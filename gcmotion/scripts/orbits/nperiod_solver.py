r"""
==============
NPeriod Solver
==============

A Custom ODE solver basing SciPy's ``OdeSolver``.

It is essentially identical to SciPy's ``RK45`` method, with a hard-coded
event, and an extra check: For every solving step, if the event triggers, it
checks if the current value of its conjugate variable is also close to its
initial value. If so, the Solver treats this as a full period of the orbit,
and can stop the solving after an arbitrary number of full periods.
"""

import warnings
import numpy as np
from math import isclose, fmod
from scipy.integrate import OdeSolver, RK45
from scipy.integrate._ivp.ivp import (
    prepare_events,
    find_active_events,
    handle_events,
)
from scipy.integrate._ivp.rk import (
    RkDenseOutput,
    rk_step,
    # Multiply steps computed from asymptotic behaviour of errors by this.
    SAFETY,
    MIN_FACTOR,  # Minimum allowed decrease in a step size.
    MAX_FACTOR,  # Maximum allowed increase in a step size.
)
from scipy.integrate._ivp.common import (
    norm,
    select_initial_step,
    validate_first_step,
)

from typing import Callable

from gcmotion.utils.logger_setup import logger
from gcmotion.scripts.events import when_theta, when_psi
from gcmotion.configuration.scripts_configuration import (
    NPeriodSolverConfig,
)

MAX_RECURSION_DEPTH = NPeriodSolverConfig.max_recursion_depth
HARDCODED_TERMINAL = np.iinfo(np.int64).max
EVENT_VARIABLE = NPeriodSolverConfig.event_variable
T_STOP = NPeriodSolverConfig.tstopNU

tau = 2 * np.pi

ESCAPED_WALL = "Halted: The particle escaped the wall."
TIMED_OUT = "Halted: The integration timed out."


class NPeriodSolver(OdeSolver):
    r"""Custom OdeSolver modeled after SciPy's RK45 Solver.

    It is essentially identical to SciPy's ``RK45`` method, with a hard-coded
    event, and an extra step: For every solving step, the event triggers when
    the corresponding variable returns to its initial value. It then checks if
    the current value of its conjugate variable is also close to its initial
    value. If so, the Solver treats this as a full period of the orbit.

    Depending on the value of the ``stop_after`` arguement, the integration can
    halt after an arbitrary number of full periods.

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

    event = when_theta if EVENT_VARIABLE == "theta" else when_psi

    def __init__(
        self,
        fun: Callable,
        t0: float,
        y0: np.ndarray,
        t_bound: float,
        first_step=None,
        rtol: float = NPeriodSolverConfig.rtol,
        atol: float = NPeriodSolverConfig.atol,
        max_step: float = NPeriodSolverConfig.max_step,
        **extraneous,
    ):
        r"""Arguements are as defined by ``OdeSolver``."""

        logger.info("==> Setting up NPeriodsSolver...")
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
        # from the period_check() check) is that we've hardcoded our event.
        # This makes the code much easier to work with, since we've removed the
        # added abstraction of handling an arbitrary number of events.
        self.theta0 = fmod(y0[0], tau)
        self.psi0 = y0[1]
        self.event_root, self.conjugate_root = (
            (self.theta0, self.psi0)
            if EVENT_VARIABLE == "theta"
            else (self.psi0, self.theta0)
        )
        event = __class__.event(
            root=self.event_root, terminal=HARDCODED_TERMINAL
        )
        logger.info(f"\tEvent used: {event.name}")

        # Prepare event
        events, self.max_events, self.event_dir = prepare_events([event])
        self.event = events[0]  # Prepared event
        self.ev_value = self.event(t0, y0)
        self.t_events = []
        self.y_events = []

        self.event_count = np.zeros(1)
        self.sol = None
        self.num_steps = 0

        self.psi_wall = extraneous["psi_wallNU"]
        # HACK:These are references to the corresponing Particle's attributes,
        # since there is no other way for the solver to pass data to Particle.
        self.stop_after = extraneous["stop_after"]
        self.t_periods = extraneous["t_periods"]
        self.y_final = extraneous["y_final"]
        self.flags = extraneous["flags"]

        self.periods_completed = 0

        # This flag becomes True when the final event has triggered, and we
        # start to recursively call step() to end the integration on the exact
        # time of the trigger.
        self.recursing = False

    def _step_impl(self):
        r"""Identical to RK45, with an extra call to period_check().

        :meta public:
        """

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
            # Halt if the particle hits the wall
            if y[1] > self.psi_wall:
                self.status = "failed"
                self.flags["escaped_wall"] = True
                self.flags["timed_out"] = False
                self.flags["succeded"] = False
                logger.warning(ESCAPED_WALL)
                warnings.warn(ESCAPED_WALL)
                return False, ESCAPED_WALL
            # Halt if the particle takes too much time
            if abs(t) > T_STOP and self.periods_completed == 0:
                self.status = "failed"
                self.flags["escaped_wall"] = False
                self.flags["timed_out"] = True
                self.flags["succeded"] = False
                logger.warning(TIMED_OUT)
                warnings.warn(TIMED_OUT)
                return False, TIMED_OUT

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
        if not self.recursing and self.period_check():
            self.status = "finished"
            return False, (
                "(NPeriods Solver) "
                f"Solver stopped after {self.periods_completed} periods."
            )
        self.ev_value = self.ev_value_new

        self.num_steps += 1

        return True, None

    def period_check(self):
        r"""Checks if the orbit is close to completing a full period, and
        calls last_step_recursion() to halt the integration after
        ``stop_after`` periods.

        This method is called inside *every* step() iteration, but only
        proceeds if the integration is 'close' to a trigger point, e.g. the
        `event variable` is closing in on its initial value, and calculates the
        exact time this happens by a root finding algorithm.

        It then checks if the `conjugate variable` is also closing in on its
        initial value. If so, the current state of integration, (t, y) is
        stored as an event (in contrast to how normal SciPy events work). The
        `periods_completed` is incremented by 1.

        If the `periods_completed` attribute has reach the `stop_after`
        parameter, it means that we are just before or just after the point we
        want to halt the integration, depending on where the last step landed
        with respect to the event point. If the solver overstepped, we take one
        step back and call last_step_recursion().

        """
        # Avoid stoping slow-starting orbits the moment they start. Also
        # dense_output() needs at least 1 completed step.
        if self.num_steps < NPeriodSolverConfig.min_step_num:
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
            if self._isclose_to_period():
                self.t_periods.append(roots[0])
                self.periods_completed += 1
                if self.periods_completed == self.stop_after:
                    self.t_goal = roots[0]
                    self.y_goal = self.y_events[-1]
                    self.last_step_recursion()
                    return True

    def last_step_recursion(self):
        r"""Take recursing steps with smaller and smaller steps to reach the
        exact end of the integration.

        This method is called only when the full number of periods have
        been completed and the solving is reaching its end. At this point, the
        period_check() method is deactivated, and step() works identically to
        the RK45 method, except that the step_size is pre-set.

        It recursively calls step() with a step_size equal to half the distance
        between of the last point and the event point. This asymptotically
        leads the orbit to its closing point, since we basically keep taking
        smaller and smaller steps to our target without overshooting.

        Note that this recursion happens within the solver itself, in contrast
        to step() which is normally called only in solve_ivp(). This means that
        the solver's dense_output is not updated, so we set the Particle's
        `y_final` attribute to the last values we calculated. Those are later
        appended to the total calculated timeseries.
        """
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
        self.flags["escaped_wall"] = False
        self.flags["timed_out"] = False
        self.flags["succeded"] = True

    def _isclose_to_period(self):
        r"""If ψ is the event variable, then we must mod theta to 2π"""

        if EVENT_VARIABLE == "theta":
            return isclose(
                self.y_events[-1][1],
                self.conjugate_root,
                rel_tol=self.stop_point_rtol,
                abs_tol=self.stop_point_atol,
            )
        elif EVENT_VARIABLE == "psi":
            return isclose(
                fmod(self.y_events[-1][0], tau),
                fmod(self.conjugate_root, tau),
                rel_tol=self.stop_point_rtol,
                abs_tol=self.stop_point_atol,
            )

    def _estimate_error(self, K, h):
        r"""Identical to RK45"""
        return np.dot(K.T, self.E) * h

    def _estimate_error_norm(self, K, h, scale):
        r"""Identical to RK45"""
        return norm(self._estimate_error(K, h) / scale)

    def _dense_output_impl(self):
        r"""Identical to RK45."""

        Q = self.K.T.dot(self.P)
        return RkDenseOutput(self.t_old, self.t, self.y_old, Q)
