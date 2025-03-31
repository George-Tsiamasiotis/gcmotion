r"""
===============
Particle Entity
===============

Defines a particle inside a specific tokamak device and of a specific profile.
its run() method solves the differential equations in NU, as described by
White, to calculate its orbit.
"""

import pint
import warnings
import numpy as np
from collections import namedtuple
from termcolor import colored
from time import time

from gcmotion.utils.logger_setup import logger
from gcmotion.utils.pprint_dict import pprint_dict
from gcmotion.scripts.orbits.calculate_orbit import calculate_orbit

from .tokamak import Tokamak
from .profile import Profile
from .initial_conditions import InitialConditions

# Quantity alias for type annotations
type Quantity = pint.UnitRegistry.Quantity

EVENTS_IGNORED = """
Warning: events are ignored when 'NPeriods' method is used.
"""
ILLEGAL_STOP_AFTER = """
Error: 'stop_after' must be a positive integer.
"""
T_EVAL_IGNORED = """
Warning: 't_eval' arguement is ignored when 'NPeriods' method is used.
"""
STOP_AFTER_IGNORED = """
Warning: 'stop_after' arguement is ignored when "RK45" method is used.
"""


class Particle:
    r"""Creates a specific Particle in a specific Profile and with specific
    InitialConditions.

    A particle entity represents a fully-fledged particle inside a specific
    tokamak device, and defined initial conditions.

    Parameters
    ----------
    tokamak : :py:class:`~gcmotion.Tokamak`
        Tokamak object containing information about the tokamak.
    init : :py:class:`~gcmotion.InitialConditions`
        InitialConditions object containing the set of initial condtions of
        the particle.

    Notes
    -----
    To view the particle's attributes, use its
    :py:meth:`~gcmotion.Particle.quantities` method.

    Examples
    --------
    Creating a Particle:

    >>> import gcmotion as gcm
    >>> import numpy as np
    >>>
    >>> # Quantity Constructor
    >>> Rnum = 1.65
    >>> anum = 0.5
    >>> B0num = 1
    >>> species = "p"
    >>> Q = gcm.QuantityConstructor(R=Rnum, a=anum, B0=B0num, species=species)
    >>>
    >>> # Intermediate Quantities
    >>> R = Q(Rnum, "meters")
    >>> a = Q(anum, "meters")
    >>> B0 = Q(B0num, "Tesla")
    >>> i = Q(10, "NUPlasma_current")
    >>> g = Q(1, "NUPlasma_current")
    >>> Ea = Q(73500, "Volts/meter")
    >>>
    >>> # Construct a Tokamak
    >>> tokamak = gcm.Tokamak(
    ...     R=R,
    ...     a=a,
    ...     qfactor=gcm.qfactor.Hypergeometric(a, B0, q0=1.1, q_wall=3.8, n=2),
    ...     bfield=gcm.bfield.LAR(B0=B0, i=i, g=g),
    ...     efield=gcm.efield.Radial(a, Ea, B0, peak=0.98, rw=1 / 50),
    ... )
    >>>
    >>> # Setup Initial Conditions for "RK45" method
    >>> init = gcm.InitialConditions(
    ...     species="p",
    ...     muB=Q(0.5, "keV"),
    ...     Pzeta0=Q(-0.015, "NUCanonical_momentum"),
    ...     theta0=0,
    ...     zeta0=0,
    ...     psi0=Q(0.6, "psi_wall"),
    ...     t_eval=Q(np.linspace(0, 1e-3, 1000), "seconds"),
    ... )
    >>>
    >>> # Create the particle
    >>> particle = gcm.Particle(tokamak=tokamak, init=init)

    Running the particle with the *RK45* method:

    >>> particle.run()
    >>>
    >>> events = [gcm.events.when_theta(init.theta0, 3)]
    >>> particle.run(events=events)

    Running the particle with the *NPeriods* method:

    >>> particle.run(method="NPeriods", stop_after=1)
    >>> particle.run(method="NPeriods", stop_after=4)

    .. note::

        The InitialCondition's ``t_eval`` arguement is ignored when the method
        `NPeriods` is used.

    """

    def __init__(
        self,
        tokamak: Tokamak,
        init: InitialConditions,
    ):
        r"""Initializes particle Quantities and Tokamak configuration."""
        logger.info(f"==> Initializing Particle: {init.species_name}...")

        # Extend initial conditions set
        logger.info("\tCalculating full set of initial conditions.")
        init._calculate_full_set(tokamak)

        # Store those for easier reference
        self.tokamak = tokamak
        self.init = init

        # Grab attributes form arguements
        self.__dict__.update(self.tokamak.__dict__)
        self.__dict__.update(self.init.__dict__)
        logger.info("\tStored tokamak's and init's attributes.")

        # Create the particle's specific profile
        self.profile = Profile(
            tokamak=self.tokamak,
            species=self.species,
            mu=self.mu,
            Pzeta=self.Pzeta0,
            E=self.E,
        )
        logger.info("\tCreated particle's specific profile object.")

        self.input_vars = vars(self).copy()  # Store __init__() vars
        logger.info(
            f"--> Particle {init.species_name} initialization complete."
        )

    def quantities(
        self,
        which: str = "",
        everything: bool = False,
    ):
        """Prints the pint Quantities of the object.

        Parameters
        ----------
        which : str, optional
            Options on which Quantities to print. Can include many options.

                #. "init" :
                    Prints all the Quantities defined upon the particle's
                    instanciation.
                #. "NU" or "SI":
                    Print the respective subset of Quantities

            Options can be chained together, for example "initNU".
            Defaults to "" (prints all *Quantites*)

        everything : bool, optional
            Whether or not to print *all* particle's *attributes*, and not just
            Quantities. Ignores "which" arguement if True. Defaults to False.

        """

        units = "NU" if "NU" in which else "SI" if "SI" in which else ""

        if "init" in which:  # TODO: maybe use regex instead
            pprint_dict(self.input_vars, everything=everything, units=units)
        else:
            pprint_dict(self.__dict__, everything=everything, units=units)

    def run(
        self,
        method: str = "RK45",
        stop_after: int = None,
        events: list = [],
        info: bool = False,
    ):
        r"""
        Calculates the particle's orbit. The results are stored in both SI and
        NU.

        Parameters
        ----------
        method: {"NPeriods", "RK45"}, optional
            The solving method, passed to SciPy's ``solve_ivp``. Options are
            "RK45" (Runge-Kutta 4th order), "NPeriods" (Stops the orbit after N
            full :math:`\psi-P_\theta` periods). Events are ignored if the
            method is "NPeriods". Defaults to "RK45"
        stop_after: int, optional
            After how many full periods to stop the solving, if the method is
            "NPeriods". Ignored if the method is "RK45". Defaults to None.
        events : list, optional
            The list of events to be passed to the solver, if the method is
            "RK45". Ignored if the method is "NPeriods". Defaults to [].
        info : bool, optional
            Whether or not to print an output message. Defaults to False.

        Notes
        -----
        See :py:class:`~gcmotion.Particle` documentation for examples

        """
        logger.info("\tParticle's 'run' routine is called.")

        self._process_run_args(method, stop_after, events)
        self.method = method
        self.stop_after = stop_after
        self.events = events

        logger.info("\tCalculating orbit in NU...")
        # Orbit Calculation
        start = time()
        solution = self._run_orbit()
        end = time()
        self.solve_time = self.Q(end - start, "seconds")
        logger.info(
            f"\tCalculation complete. Took {
                self.solve_time:.4g~#P}."
        )

        self.theta = self.Q(solution.theta, "radians")
        self.zeta = self.Q(solution.zeta, "radians")
        self.psiNU = self.Q(solution.psi, "NUMagnetic_flux")
        self.rhoNU = self.Q(solution.rho, "NUmeters")
        self.psipNU = self.Q(solution.psip, "NUMagnetic_flux")
        self.PthetaNU = self.Q(solution.Ptheta, "NUCanonical_momentum")
        self.PzetaNU = self.Q(solution.Pzeta, "NUCanonical_momentum")
        self.t_solveNU = self.Q(solution.t_eval, "NUseconds")
        self.t_eventsNU = self.Q(solution.t_events, "NUseconds")
        self.y_events = solution.y_events
        self.message = solution.message
        logger.info(f"\tSolver message: '{self.message}'")

        # Converting back to SI
        logger.info("\tConverting results to SI...")
        start = time()
        self.psi = self.psiNU.to("Magnetic_flux")
        self.rho = self.rhoNU.to("meters")
        self.psip = self.psipNU.to("Magnetic_flux")
        self.Ptheta = self.PthetaNU.to("Canonical_momentum")
        self.Pzeta = self.PzetaNU.to("Canonical_momentum")
        self.t_solve = self.t_solveNU.to("seconds")
        self.t_events = self.t_eventsNU.to("seconds")
        end = time()
        self.conversion_time = self.Q(end - start, "seconds")

        logger.info(
            f"\tConversion completed. Took {
                self.conversion_time:.4g~#P}."
        )

        if self.method == "NPeriods":
            self._calculate_frequencies()

        self.solver_output = self._solver_output_str()

        if info:
            print(self.__str__())

    def _process_run_args(self, method: str, stop_after: int, events: list):
        r"""Processes run()'s args before passed to the solver."""

        if method == "RK45":
            logger.info("\tSolver method: RK45")
            if events == []:
                logger.info("\tRunning without events.")
            else:
                logger.info(f"\tActive events: {[x.name for x in events]}")
            if stop_after is not None:
                logger.warning(STOP_AFTER_IGNORED)
                warnings.warn(STOP_AFTER_IGNORED)

        elif method == "NPeriods":
            logger.info(f"\tSolver method: NPeriods, stop after {stop_after}.")
            if self.t_eval is not None:
                logger.warning(T_EVAL_IGNORED)
                warnings.warn(T_EVAL_IGNORED)
            if events != []:
                logger.warning(EVENTS_IGNORED)
                warnings.warn(EVENTS_IGNORED)

            if (
                (stop_after is None)
                or (not isinstance(stop_after, int))
                or (stop_after < 1)
            ):
                logger.error(ILLEGAL_STOP_AFTER)
                raise ValueError(ILLEGAL_STOP_AFTER)

    def _solver_output_str(self):

        output_str = (
            colored("\nSolver output\n", "red")
            + f"{'Solver message':>23} : {self.message:<16}\n"
        )

        if self.method == "RK45":
            # Percentage of t_eval
            tfinal = self.t_eval[-1]
            self.orbit_percentage = float(100 * (self.t_solve[-1] / tfinal).m)
            logger.info(
                f"'t_eval' percentage calculated: {self.orbit_percentage:.4g}%"
            )

            output_str += (
                f"{'Percentage':>23} : " f"{self.orbit_percentage:.1f}%\n"
            )

        elif self.method == "NPeriods":
            frequency_str = (
                colored("\nFrequencies:\n", "red")
                + f"{'Poloidal Frequency ωθ':>23} : "
                f"{f'{self.omega_theta : .4g#~}':<16}"
                f"({self.omega_thetaNU.m :.4g} [ω0])\n"
                f"{'Toroidal Frequency ωζ':>23} : "
                f"{f'{self.omega_zeta :.4g#~}':<16}"
                f"({self.omega_zetaNU.m :.4g} [ω0])\n"
                f"{'qkinetic':>23} : {f'{self.qkinetic :.4g}':<16}"
            )

        output_str += (
            f"{'Orbit calculation time':>23} : "
            f"{self.solve_time:.4g~#P}\n"
            f"{'Conversion to SI time':>23} : "
            f"{self.conversion_time:.4g~#P}\n"
        )

        if self.method == "NPeriods":
            output_str += frequency_str

        return output_str

    def _run_orbit(self):
        """Groups the particle's initial conditions and passes them to the
        solver script :py:mod:`~gcmotion.scripts.orbit`. The unpacking takes
        place in :py:meth:`~gcmotion.Particle.run`.

        Returns
        -------
        namedtuple
            The solution tuple returned by the solver.

        """
        OrbitParameters = namedtuple(
            "Orbit_Parameters", ["theta0", "psi0", "zeta0", "rho0", "mu", "t"]
        )

        parameters = OrbitParameters(
            theta0=self.theta0,
            zeta0=self.zeta0,
            psi0=self.psi0NU.magnitude,
            rho0=self.rho0NU.magnitude,
            mu=self.muNU.magnitude,
            t=self.t_evalNU.magnitude if self.method == "RK45" else None,
        )

        # HACK: The solver really has no way of interacting with the Particle
        # Class directly, but we can exploit Python's "Pass by Object
        # Reference" by passing a mutable list and let the solver fill it.
        self.t_periods = []
        return calculate_orbit(
            parameters=parameters,
            profile=self.profile,
            method=self.method,
            events=self.events,
            stop_after=self.stop_after,
            t_periods=self.t_periods,
        )

    def _calculate_frequencies(self):
        r"""Calculates the particle's ωθ, ωζ and qkinetic, if the orbit was
        calculated with the 'NPeriods' method.
        """

        if self.stop_after == 1:
            _T_theta = self.t_periods[0]
        else:
            _T_theta = np.mean(np.diff(self.t_periods))

        _omega_theta = 2 * np.pi / _T_theta
        _delta_zeta = self.zeta[-1].m - self.zeta[0].m
        _omega_zeta = _delta_zeta / _T_theta
        _qkinetic = _omega_zeta / _omega_theta

        self.omega_thetaNU = self.Q(_omega_theta, "NUw0")
        self.omega_zetaNU = self.Q(_omega_zeta, "NUw0")
        self.qkinetic = _qkinetic

        self.omega_theta = self.omega_thetaNU.to("Hertz")
        self.omega_zeta = self.omega_zetaNU.to("Hertz")

    def __str__(self):
        string = self.tokamak.__str__()
        string += self.init.__str__()
        string += getattr(self, "solver_output", "")
        return string

    def __repr__(self):
        string = self.tokamak.__repr__()
        string += self.init.__repr__()
        return string
