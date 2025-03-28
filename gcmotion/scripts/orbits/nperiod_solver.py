r"""
==============
NPeriod Solver
==============

A Custom ODE solver basing SciPy's ``OdeSolver``.

It is essentially identical to SciPy's ``RK45`` method, with a hard-coded
``when_theta`` event, and an extra check: For every solving step, if
`when_theta` triggers, it checks if the current :math:`\psi` value is also
close to the particle's initial :math:`\psi_0`. If so, the Solver treats this
as a full period of the orbit, and can stop the solving after an arbitrary
number of full periods, assuming

"""

import numpy as np
from scipy.integrate import OdeSolver, RK45
from scipy.integrate._ivp.ivp import (
    prepare_events,
    find_active_events,
    handle_events,
)
from scipy.integrate._ivp.rk import (
    RkDenseOutput,
    rk_step,
    SAFETY,
    MIN_FACTOR,
    MAX_FACTOR,
)
from scipy.integrate._ivp.common import (
    norm,
    select_initial_step,
    validate_first_step,
)

from numpy.typing import ArrayLike
from typing import Callable
from math import isclose

from gcmotion.scripts.events import when_theta
from gcmotion.configuration.scripts_configuration import (
    NPeriodSolverConfig,
)

MAX_RECURSION_DEPTH = NPeriodSolverConfig.max_recursion_depth


class NPeriodSolver(OdeSolver):
    r"""Custom OdeSolver modeled after SciPy's RK45 Solver.

    The coefficients belong to the `Dormand-Prince method
    <https://en.wikipedia.org/wiki/Dormand%E2%80%93Prince_method>`_, which is a
    member of the Runge-Kutta family.

    References
    ----------
    .. [1] J. R. Dormand, P. J. Prince, "A family of embedded Runge-Kutta
           formulae", Journal of Computational and Applied Mathematics, Vol. 6,
           No. 1, pp. 19-26, 1980.
    .. [2] L. W. Shampine, "Some Practical Runge-Kutta Formulas", Mathematics
           of Computation,, Vol. 46, No. 173, pp. 135-150, 1986.
    """

    # RK45 Class Attributes (Coefficients and orders)
    C: np.ndarray = RK45.C  # Butcher Tableu Coefficients
    A: np.ndarray = RK45.A  # Butcher Tableu Coefficients
    B: np.ndarray = RK45.B  # Butcher Tableu Coefficients
    E: np.ndarray = RK45.E  # Used for error estimation
    # Used in _dense_output_impl(), which is called by solve_ivp after the
    # integration is complete.
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
        rtol: float = NPeriodSolverConfig.rtol,
        atol: float = NPeriodSolverConfig.atol,
        max_step: float = NPeriodSolverConfig.max_step,
        **extraneous,
    ):

        super().__init__(
            fun, t0, y0, t_bound, vectorized=False, support_complex=False
        )

        self.y_old = None
        self.max_step = max_step
        self.rtol, self.atol = rtol, atol
        self.stop_point_rtol = NPeriodSolverConfig.stop_point_rtol
        self.stop_point_atol = NPeriodSolverConfig.stop_point_atol
        # Those are initialised in super(), self.t=t0, self.y=y0
        self.f = self.fun(self.t, self.y)

        # "Empirically" selects a good first step
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

        # Storage array for putting RK stages
        self.K = np.empty((self.n_stages + 1, self.n), dtype=self.y.dtype)
        self.error_exponent = -1 / (self.error_estimator_order + 1)
        self.h_previous = None

        # PERF: From now on, step() and event location work the same as RK45
        # and solve_ivp respectively. The only difference in our solver (apart
        # from the _stop() check) is that we've hardcoded our event. This makes
        # the code much easier to work with, since we've removed the added
        # abstraction of handling an arbitrary number of events.
        self.theta0 = y0[0]
        self.psi0 = y0[1]
        event = when_theta(root=self.theta0, terminal=100000)
        events, self.max_events, self.event_dir = prepare_events([event])
        self.event = events[0]
        self.ev_value = self.event(t0, y0)
        self.t_events = []
        self.y_events = []

        self.event_count = np.zeros(1)
        self.sol = None
        self.num_steps = 0
        self.stop_after = extraneous["stop_after"]
        self.t_periods = extraneous["t_periods"]
        self.y_final = extraneous["y_final"]
        self.periods_completed = 0
        self.last_step = False
        self.recursing = False

    def _estimate_error(self, K, h):
        return np.dot(K.T, self.E) * h

    def _estimate_error_norm(self, K, h, scale):
        return norm(self._estimate_error(K, h) / scale)

    def _stop(self):

        # self.dense_output() needs at least one completed step
        if self.num_steps == 0:
            return

        active_events = find_active_events(
            [self.ev_value], [self.ev_value_new], self.event_dir
        )
        if active_events.size > 0:

            sol = self.dense_output()

            self.event_count[active_events] += 1
            root_indices, roots, terminate = handle_events(
                sol=sol,
                events=[self.event],
                active_events=active_events,
                event_count=self.event_count,
                max_events=self.max_events,
                t_old=self.t_old,
                t=self.t,
            )
            for e, te in zip(root_indices, roots):
                self.t_events.append(te)
                self.y_events.append(sol(te))

            # When the particle
            if isclose(
                self.y_events[-1][1],
                self.psi0,
                rel_tol=self.stop_point_rtol,
                abs_tol=self.stop_point_atol,
            ):
                self.t_periods.append(roots[0])
                self.periods_completed += 1
                if self.periods_completed == self.stop_after:
                    self.t_goal = roots[0]
                    self.y_goal = self.y_events[-1]
                    self._recurse_last_event()
                    return True

    def _recurse_last_event(self):
        self.recursing = True

        if self.t > self.t_goal:  # Overstepped
            self.t = self.t_old
            self.y = self.y_old

        count = 0

        while (
            not np.all(
                np.isclose(
                    self.y,
                    self.y_goal,
                    NPeriodSolverConfig.final_y_rtol,
                )
            )
            and count <= MAX_RECURSION_DEPTH
        ):
            self.h_abs = abs(self.t - self.t_goal) / 2
            self.step()
            count += 1

        self.y_final.append(self.y)

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

        self.ev_value_new = self.event(self.t, self.y)
        if not self.recursing and self._stop():
            self.status = "finished"
            return False, (
                "(NPeriods Solver) "
                f"Solver stopped after {self.stop_after} periods."
            )

        self.ev_value = self.ev_value_new

        self.num_steps += 1

        return True, None

    def _dense_output_impl(self):
        Q = self.K.T.dot(self.P)
        return RkDenseOutput(self.t_old, self.t, self.y_old, Q)
