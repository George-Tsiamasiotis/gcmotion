import numpy as np
from scipy.integrate import OdeSolver, RK45
from scipy.integrate._ivp.rk import RkDenseOutput, rk_step
from scipy.integrate._ivp.common import (
    norm,
    select_initial_step,
    validate_first_step,
)

from numpy.typing import ArrayLike
from typing import Callable

from gcmotion.configuration.scripts_configuration import (
    SinglePeriodSolverConfig,
)


# Multiply steps computed from asymptotic behaviour of errors by this.
SAFETY = 0.9

MIN_FACTOR = 0.2  # Minimum allowed decrease in a step size.
MAX_FACTOR = 10  # Maximum allowed increase in a step size.


class SinglePeriodSolver(OdeSolver):

    # Runge-Kutta Class Attributes (Coefficients and orders)
    C: np.ndarray = RK45.C
    A: np.ndarray = RK45.A
    B: np.ndarray = RK45.B
    E: np.ndarray = RK45.E
    P: np.ndarray = RK45.P
    order: int = RK45.order
    error_estimator_order: int = RK45.error_estimator_order
    n_stages: int = RK45.n_stages

    def __init__(
        self,
        fun: Callable,
        t0: float,
        y0: ArrayLike,
        t_bound: float,
        first_step=None,
        rtol: float = SinglePeriodSolverConfig.rtol,
        atol: float = SinglePeriodSolverConfig.atol,
        max_step: float = SinglePeriodSolverConfig.max_step,
        vectorized: bool = SinglePeriodSolverConfig.vectorized,
        **extraneous,
    ):

        super().__init__(
            fun, t0, y0, t_bound, vectorized, support_complex=False
        )
        self.y_old = None
        self.max_step = max_step
        self.rtol, self.atol = rtol, atol
        self.f = self.fun(self.t, self.y)

        # first step size
        if first_step is None:
            self.h_abs = select_initial_step(
                self.fun,
                self.t,
                self.y,
                t_bound,
                max_step,
                self.f,
                self.direction,
                self.error_estimator_order,
                self.rtol,
                self.atol,
            )
        else:
            self.h_abs = validate_first_step(first_step, t0, t_bound)
        self.K = np.empty((self.n_stages + 1, self.n), dtype=self.y.dtype)
        self.error_exponent = -1 / (self.error_estimator_order + 1)
        self.h_previous = None

    def _estimate_error(self, K, h):
        return np.dot(K.T, self.E) * h

    def _estimate_error_norm(self, K, h, scale):
        return norm(self._estimate_error(K, h) / scale)

    def _step_impl(self):

        t = self.t
        y = self.y

        max_step = self.max_step
        rtol = self.rtol
        atol = self.atol

        min_step = 10 * np.abs(np.nextafter(t, self.direction * np.inf) - t)

        if self.h_abs > max_step:
            h_abs = max_step
        elif self.h_abs < min_step:
            h_abs = min_step
        else:
            h_abs = self.h_abs

        step_accepted = False
        step_rejected = False

        while not step_accepted:
            # Calculate step size
            if h_abs < min_step:
                return False, self.TOO_SMALL_STEP

            h = h_abs * self.direction
            t_new = t + h

            if self.direction * (t_new - self.t_bound) > 0:
                t_new = self.t_bound

            h = t_new - t
            h_abs = np.abs(h)

            # y_new = solution at t+h
            # f_new = derivative at t+h and y_new
            y_new, f_new = rk_step(
                self.fun, t, y, self.f, h, self.A, self.B, self.C, self.K
            )
            # ?????
            scale = atol + np.maximum(np.abs(y), np.abs(y_new)) * rtol
            error_norm = self._estimate_error_norm(self.K, h, scale)

            if error_norm < 1:
                if error_norm == 0:
                    factor = MAX_FACTOR
                else:
                    factor = min(
                        MAX_FACTOR, SAFETY * error_norm**self.error_exponent
                    )

                if step_rejected:
                    factor = min(1, factor)

                h_abs *= factor

                step_accepted = True
            else:
                h_abs *= max(
                    MIN_FACTOR, SAFETY * error_norm**self.error_exponent
                )
                step_rejected = True

        self.h_previous = h
        self.y_old = y
        self.t = t_new
        self.y = y_new
        self.h_abs = h_abs
        self.f = f_new

        return True, None

    def _dense_output_impl(self):
        Q = self.K.T.dot(self.P)
        return RkDenseOutput(self.t_old, self.t, self.y_old, Q)
