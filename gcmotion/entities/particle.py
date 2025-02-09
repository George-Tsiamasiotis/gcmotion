r"""
===============
Particle Entity
===============

Defines a particle inside a specific tokamak device and of a specific profile.
its run() method solves the differential equations in NU, as described by
White, to calculate its orbit.
"""

import pint
import numpy as np
from collections import namedtuple
from termcolor import colored
from time import time

from gcmotion.utils.logger_setup import logger
from gcmotion.utils.pprint_dict import pprint_dict
from gcmotion.scripts.orbit import orbit

from gcmotion.entities.tokamak import Tokamak
from gcmotion.entities.profile import Profile
from gcmotion.entities.initial_conditions import InitialConditions

# Quantity alias for type annotations
type Quantity = pint.UnitRegistry.Quantity


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
    Here is how a particle is created:


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
    >>> # Setup Initial Conditions
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
    >>> # Create the particle and calculate its obrit
    >>> particle = gcm.Particle(tokamak=tokamak, init=init)
    >>> particle.run()

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
        orbit=True,
        info: bool = False,
        events: list = [],
    ):
        r"""
        Calculates the particle's orbit. The results are stored in both SI and
        NU.

        Parameters
        ----------
        info : bool, optional
            Whether or not to print an output message. Defaults to False.
        events : list, optional
            The list of events to be passed to the solver. Defaults to [ ].

        """
        logger.info("\tParticle's 'run' routine is called.")

        if events == []:
            logger.info("\tRunning without events.")
        else:
            logger.info(f"\tActive events: {[x.name for x in events]}")

        logger.info("\tCalculating orbit in NU...")
        # Orbit Calculation
        start = time()
        solution = self._orbit(events=events)
        end = time()
        solve_time = self.Q(end - start, "seconds")
        logger.info(
            f"\tCalculation complete. Took {
                solve_time:.4g~#P}."
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
        message = solution.message
        logger.info(f"\tSolver message: '{message}'")

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
        conversion_time = self.Q(end - start, "seconds")

        logger.info(
            f"\tConversion completed. Took {
                conversion_time:.4g~#P}."
        )

        # Percentage of t_eval
        tfinal = self.t_eval[-1]
        self.orbit_percentage = float(100 * (self.t_solve[-1] / tfinal).m)
        logger.info(
            f"'t_eval' percentage calculated: {self.orbit_percentage:.4g}%"
        )
        logger.trace(f"{self.theta0=}, {self.theta[-1].m % (2*3.1415)=}")
        logger.trace(f"t_events = {self.t_events}")
        logger.trace(
            f"len(t_events)= {len(np.array(self.t_events.m).flatten())}"
        )

        self.solver_output = (
            colored("\nSolver output: ", "red") + f"{message}\n"
            f"{'Percentage':>23} : "
            f"{self.orbit_percentage:.1f}%\n"
            f"{'Orbit calculation time':>23} : "
            f"{solve_time:.4g~#P}\n"
            f"{'Conversion to SI time':>23} : "
            f"{conversion_time:.4g~#P}\n"
        )

        if info:
            print(self.__str__())

    def _orbit(self, events: list = []):
        """Groups the particle's initial conditions and passes them to the
        solver script :py:mod:`~gcmotion.scripts.orbit`. The unpacking takes
        place in :py:meth:`~gcmotion.Particle.run`.

        Parameters
        ----------
        events : list, optional
            List containing the :py:mod:`~gcmotion.scripts.events`. Defaults to
            []

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
            t=self.t_evalNU.magnitude,
            mu=self.muNU.magnitude,
        )

        profile = self.profile

        return orbit(parameters, profile, events=events)

    def __str__(self):
        string = self.tokamak.__str__()
        string += self.init.__str__()
        string += getattr(self, "solver_output", "")
        return string

    def __repr__(self):
        string = self.tokamak.__repr__()
        string += self.init.__repr__()
        return string
