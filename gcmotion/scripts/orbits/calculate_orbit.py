r"""
=====
Orbit
=====

Simple function wrapper around SciPy's solve_ivp() solver. It solves White's
differential equations in NU.

It uses the RK4(5) solving method. The tolerance can be tweaked in
gcmotion/configuration/solver_configuration.py.

The solver only calculates the time evolution of theta, psi, zeta and rho. The
rest of the dynamical variables are calculated afterwards.

Important
---------
The solver expects the input to be purely numerical quantites, in [NU], and
returns the solution also in [NU]. The conversion to SI units is handled by the
particle's run() method.

Tip
---
When setting up the analytical function dSdt to pass to the solver, it is
better to use Python's built-in mathematical functions by importing the math
module, instead of their numpy counterparts. Python's functions are much faster
than numpy's when calculating single values (often by a factor of 10 or more)..

"""

import numpy as np
from scipy.integrate import solve_ivp
from collections import namedtuple

from gcmotion.entities.profile import Profile
from .nperiod_solver import NPeriodSolver

from gcmotion.configuration.scripts_configuration import (
    NPeriodSolverConfig as config,
)

# Returned object
Solution = namedtuple(
    "Orbit_solutionNU",
    [
        "theta",
        "psi",
        "zeta",
        "rho",
        "psip",
        "Ptheta",
        "Pzeta",
        "t_eval",
        "t_events",
        "y_events",
        "message",
    ],
)


def calculate_orbit(
    parameters: namedtuple,
    profile: Profile,
    method: str,
    events: list,
    stop_after: int = None,
    t_periods: list[()] = None,
    psi_wallNU: float = np.inf,
    flags: dict = {},
) -> namedtuple:
    r"""Wrapper function around SciPy's solve_ivp().

    Parameters
    ----------
    parameters : namedtuple
        Named tuple containing the initial conditions and constants in [NU]:
        theta_0, psi_0, zeta_0, rho_0, as well as mu and t.
    profile : namedtuple
        Named tuple containing the tokamak configuration objects.
    method: {"nperiods", "rk45"}
        The solving method, passed to SciPy's ``solve_ivp``. Options are
        "RK45" (Runge-Kutta 4th order), "NPeriods" (Stops the orbit after N
        full :math:`\psi-P_\theta` periods). Events are ignored if the
        method is "NPeriods". Defaults to "NPeriods"
    stop_after: int, optional
        After how many full periods to stop the solving, if the method is
        "NPeriods". Defaults to None (RK45).
    events : list
        List containing the independed events to track. Defaults to "SI". Note
        that multiple events trigger independently from one another.
    t_periods: empty list, optional
        Reference to a Particle's attribute that stores the times of the
        calculated periods. Only used with "NPeriods" method. This is
        necessarry due to the fact that the custom Solver cannot directly
        communicate with the Particle Class. Defaults to None (RK45).
    psi_wallNU: float, optional
        The ψ value at the wall. Only used by the NPeriods solver to halt the
        integration in cases the particle escapes the wall.
    flags: dict, optional
        Dict containg flags about the solver termination, to be passed back to
        the Particle.

    Returns
    -------
    solution : namedtuple
        A named tuple containing the following:

        -> The calculated solutions of theta, psi, zeta, rho, psi_p, P_theta,
            P_zeta as np.ndarrays in [NU].

        -> A np.ndarray containing the evaluation times in [NU]. This is useful
            in the case a terminating event is used, since the orbit is
            terminated earlier.

        -> 2 lists containing the time positions and values of the found events
            in [NU]. If more than 1 event is used, the lists are nested.

        #. The solver status message.

    """

    # Parameters
    mu = parameters.mu
    t = parameters.t

    # Initial Conditions
    S0 = [
        parameters.theta0,
        parameters.psi0,
        parameters.zeta0,
        parameters.rho0,
    ]

    # Tokamak profile
    qfactor = profile.qfactor
    bfield = profile.bfield
    efield = profile.efield

    # Define quantites for the solver for clarity
    solverqNU = qfactor.solverqNU
    solverbNU = bfield.solverbNU
    solverPhiderNU = efield.solverPhiderNU

    def dSdt(t, S):
        """Sets the diff equations system to pass to scipy.

        All values are in normalized units [NU].
        """

        theta, psi, z, rho = S

        # Object methods calls
        q = solverqNU(psi)
        b, b_der, currents, currents_der = solverbNU(psi, theta)
        phi_der_psi, phi_der_theta = solverPhiderNU(psi, theta)

        # Unpack
        b_der_psi, b_der_theta = b_der
        i, g = currents
        i_der, g_der = currents_der
        # Multiply current derivatives by q to get their derivatives with
        # respect to psi instead of psip
        i_der, g_der = q * i_der, q * g_der

        # Intermediate values
        par = mu + rho**2 * b
        bracket1 = par * b_der_psi + phi_der_psi
        bracket2 = par * b_der_theta + phi_der_theta
        D = g * q + i + rho * (g * i_der - i * g_der)

        # Canonical Equations
        theta_dot = (1 - rho * g_der) / D * rho * b**2 + q * g / D * bracket1
        psi_dot = -q * g / D * bracket2
        rho_dot = -(1 - rho * g_der) / D * bracket2
        z_dot = (q + rho * i_der) / D * rho * b**2 - q * i / D * bracket1

        return [theta_dot, psi_dot, z_dot, rho_dot]

    if method == "RK45":
        method = "RK45"
        t_eval = t
        t_span = (t[0], t[-1])
        extra_options = {}
    elif method == "NPeriods":
        method = NPeriodSolver
        t_eval = None
        t_span = (0, np.inf)
        extra_options = {
            "stop_after": stop_after,
            "t_periods": t_periods,
            "y_final": [],
            "psi_wallNU": psi_wallNU,
            "flags": flags,
        }

    sol = solve_ivp(
        fun=dSdt,
        t_span=t_span,
        y0=S0,
        t_eval=t_eval,
        atol=config.atol,
        rtol=config.rtol,
        events=events,
        method=method,
        dense_output=False,
        **extra_options,
    )

    t_solve = sol.t
    t_events = sol.t_events
    y_events = sol.y_events
    message = sol.message

    if method == "RK45":
        theta = sol.y[0]
        psi = sol.y[1]
        zeta = sol.y[2]
        rho = sol.y[3]
    # Add the last y points calculated from the final recursion
    elif method is NPeriodSolver and extra_options["flags"]["succeded"]:
        y_final = extra_options["y_final"][0]
        theta = np.append(sol.y[0], y_final[0])
        psi = np.append(sol.y[1], y_final[1])
        zeta = np.append(sol.y[2], y_final[2])
        rho = np.append(sol.y[3], y_final[3])
    else:
        # In this case, the message cannot be return, so we need to add it
        # manually inside Particle.
        return

    # Calculate psip and Canonical Momenta
    _, i, g = bfield.bigNU(psi, theta)
    psip = qfactor.psipNU(psi)
    Ptheta = psi + rho * i
    Pzeta = rho * g - psip

    # Pack results and return them
    return Solution(
        theta=theta,
        psi=psi,
        zeta=zeta,
        rho=rho,
        psip=psip,
        Ptheta=Ptheta,
        Pzeta=Pzeta,
        t_eval=t_solve,
        t_events=t_events,
        y_events=y_events,
        message=message,
    )
