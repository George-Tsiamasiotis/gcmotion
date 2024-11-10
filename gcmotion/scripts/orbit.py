r"""
Simple function wrapper around SciPy's solve_ivp() solver. It solves
the dynamical system:

.. todo::
    link to a markdown file with the system's formula.

It uses the RK4(5) solving method. The tolerance can be tweaked 
in gcmotion/configuration/solver_configuration.py.

The solver only calculates the time evolution of
:math:`\theta, \psi, \zeta, \rho_{||}`. The rest of the dynamical variables
are calculated afterwards.

.. important::
    
    The solver expects the input to be purely numerical quantites, in [**NU**],
    and returns the solution also in [**NU**]. The conversion to SI units is handled
    by the particle's :py:meth:`~gcmotion.classes.particle.Particle.run` method.

.. tip::
    When setting up the analytical function dSdt to pass to the solver,
    it is better to use Python's built-in mathematical functions by importing
    the ``math`` module, instead of their ``numpy``
    counterparts. Python's functions are much faster than numpy's when calculating
    single values (often by a factor of 10 or more).

Example
-------

This is how ``orbit`` is called inside the class :py:class:`Particle`:

.. code-block:: python
    
    parameters = Parameters(
        theta0 = self.theta0.magnitude,
        psi0   = self.psi0NU.magnitude,            
        zeta0  = self.zeta0.magnitude,
        rho0   = self.rho0NU.magnitude,
        mu     = self.muNU.magnitude,
        t      = self.t_evalNU.magnitude
    )

    profile = Profile(
        qfactor = self.qfactor,
        bfield  = self.bfield,
        efield  = self.efield,
    )

    return orbit(parameters, profile, events=events)

And this is how the solution is unpacked:

.. code-block:: python

    self.theta      = self.Q(solution.theta, "radians")
    self.zeta       = self.Q(solution.zeta, "radians")
    self.psiNU      = self.Q(solution.psi, "NUMagnetic_flux")
    self.rhoNU      = self.Q(solution.rho, "NUmeters")
    self.psipNU     = self.Q(solution.psip, "NUMagnetic_flux")
    self.PthetaNU   = self.Q(solution.Ptheta, "NUMagnetic_flux")
    self.PzetaNU    = self.Q(solution.Pzeta, "NUMagnetic_flux")
    self.t_evalNU   = self.Q(solution.t_eval, "NUseconds")
    self.t_eventsNU = self.Q(solution.t_events, "NUseconds")
    self.y_events   = solution.y_events
    message         = solution.message


.. rubric:: Function:
    :heading-level: 4
"""

from scipy.integrate import solve_ivp
from collections import namedtuple

from gcmotion.utils.logger_setup import logger

from gcmotion.configuration.solver_configuration import (
    solver_configuration as config,
)


def orbit(
    parameters: namedtuple,
    profile: namedtuple,
    events: list = [],
) -> namedtuple:
    r"""Wrapper function around SciPy's solve_ivp().

    Parameters
    ----------
    parameters : namedtuple
        Named tuple containing the initial conditions and constants in [NU]:
        :math:`\theta_0, \psi_0, \zeta_0, \rho_{||,0}`, as well as
        :math:`\mu` and :math:`t`.
    profile : namedtuple
        Named tuple containing the tokamak configuration objects.
    events : list, optional
        List containing the independed events to track. Defaults to "SI"

    Returns
    -------
    solution : namedtuple
        A named tuple containing the following:

            #. The calculated solutions of
                :math:`\theta, \psi, \zeta, \rho, \psi_p, P_\theta, P_\zeta`
                as np.ndarrays in [NU].

            #. A np.ndarray containing the evaluation times in [NU].
                This is useful in the case a terminating event is used, since the
                orbit is terminated earlier.

            #. 2 lists containing the time positions and values of the found events in [NU].
                If more than 1 event is used, the lists are nested.

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
    i = bfield.iNU.magnitude
    g = bfield.gNU.magnitude
    solverqNU = qfactor.solverqNU
    solverbNU = bfield.solverbNU
    solverPhiderNU = efield.solverPhiderNU

    logger.debug(f"\tSolver: Using i={bfield.iNU:.4g~P}, g={bfield.gNU:.4g~P}")  # fmt: skip
    logger.debug(f"\tSolver: (t0, tf, steps)=({t[0]:.4g}, {t[-1]:.4g}, {len(t)})")  # fmt: skip
    logger.debug(f"\tSolver: theta0={S0[0]:.4g}, psi0={S0[1]:.4g}, zeta0={S0[2]:.4g}, rho0={S0[3]:.4g}")  # fmt: skip
    logger.debug(f"\tSolver: mu={mu:.4g}")  # fmt: skip

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
        bracket1 = -par * b_der_psi + phi_der_psi
        bracket2 = par * b_der_theta + phi_der_theta
        D = g * q + i + rho * (g * i_der - i * g_der)

        # Canonical Equations
        theta_dot = (1 - rho * g_der) / D * rho * b**2 + q * g / D * bracket1
        psi_dot = -q * g / D * bracket2
        rho_dot = -(1 - rho * g_der) / D * bracket2
        z_dot = (q + rho * i_der) / D * rho * b**2 - q * i / D * bracket1

        return [theta_dot, psi_dot, z_dot, rho_dot]

    t_span = (t[0], t[-1])
    sol = solve_ivp(
        fun=dSdt,
        t_span=t_span,
        y0=S0,
        t_eval=t,
        atol=config["atol"],
        rtol=config["rtol"],
        events=events,
        dense_output=True,
    )

    theta = sol.y[0]
    psi = sol.y[1]
    zeta = sol.y[2]
    rho = sol.y[3]
    t_eval = sol.t
    t_events = sol.t_events
    y_events = sol.y_events
    message = f"{sol.status}: {sol.message}"

    # Calculate psip and Canonical Momenta
    _, i, g = bfield.bigNU(psi, theta)
    psip = qfactor.psipNU(psi)
    Ptheta = psi + rho * i
    Pzeta = rho * g - psip

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

    # fmt: off
    return Solution(
        theta    = theta,
        psi      = psi,
        zeta     = zeta,
        rho      = rho,
        psip     = psip,
        Ptheta   = Ptheta,
        Pzeta    = Pzeta,
        t_eval   = t_eval,
        t_events = t_events,
        y_events = y_events,
        message  = message,
    )
    # fmt: on
