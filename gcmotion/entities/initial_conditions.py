r"""
========================
InitialConditions Entity
========================

This module defines the "InitialConditions" entity. The particle's initial
conditions are defined in 2 steps, since some Quantities require a Tokamak
object to be defined

"""

import pint
from termcolor import colored
from typing import Literal

from gcmotion.entities.tokamak import Tokamak
from gcmotion.configuration.physical_constants import PhysicalConstants

from gcmotion.utils.logger_setup import logger

type SupportedSpecies = Literal["p", "e", "D", "T", "He3", "He4"]
type Quantity = pint.Quantity
type QuantityArray = pint.Quantity


class InitialConditions:
    r"""Creates a set of initial conditions for a particle

    Upon initialization, only the passed arguments are setup. When creating a
    particle, the Tokamak instance is is used internally to calculate the full
    set of initial conditions, such as :math:`\rho_0`, :math:`\psi_{p0}`, ...,
    as well as the particle's energy.

    Parameters
    ----------
    species : {'p', 'e', 'D', 'T', 'He3', 'He4'}
        The particle's species. This field is case-insensitive.
    theta0 : float | Quantity
        The particle's initial :math:`\theta_0`.
    zeta0 : float | Quantity
        The particle's initial :math:`\zeta_0`.
    psi0: Quantity
        The :math:`\psi_0` initial values in dimensions of [Magnetic_flux] or
        [psi_wall]. [psi_wall] is a unit of Magnetic flux, where 1[psi_wall] is
        defined to be the Magnetic flux of the last closed surface. This way we
        can set the :math:`\psi_0` initial value with respect to the tokamak's
        :math:`\psi_{wall}`.
    t_eval : Quantity (1D array)
        The time interval return values, [:math:`t_0, t_f, steps`],
        in dimensions of [time].
    muB : Quantity
        This parameter will be parsed differently depending on its
        dimensionality. If its dimensions are [energy], then it is parsed as
        the particle's initial "perpandicular energy", :math:`\mu B`. If its
        dimensions are [current][area] i.e. dimensions of magnetic moment, it
        is parsed as the particle's magnetic moment, which is a Constant of
        motion.
    Pzeta0 : Quantity
        The particle's initial :math:`P_{\zeta 0}`.

    Examples
    --------
    How to initialize an `InitialConditions` object.

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
    >>> init = gcm.InitialConditions(
    ...     species="p",
    ...     theta0=0,
    ...     zeta0=0,
    ...     psi0=Q(0.6, "psi_wall"),
    ...     muB=Q(0.5, "keV"),
    ...     Pzeta0=Q(-0.015, "NUCanonical_momentum"),
    ...     t_eval=Q(np.linspace(0, 1e-3, 1000), "seconds"),
    ... )

    """

    def __init__(
        self,
        species: SupportedSpecies,
        theta0: float | Quantity,
        zeta0: float | Quantity,
        psi0: Quantity,
        muB: Quantity,
        Pzeta0: Quantity,
        t_eval: QuantityArray,
    ):
        r"""Initializes parameters and corresponding NU Quantities"""
        logger.info("==> Initializing 1st step InitialConditions...")

        self.Q = type(Pzeta0)
        self.species = species.lower()
        self.species_name = getattr(PhysicalConstants, self.species + "_name")
        # Grab particle's mass and charge
        M = getattr(PhysicalConstants, self.species + "_M")
        Z = getattr(PhysicalConstants, self.species + "_Z")

        self.miNU = self.Q(M, "Proton_mass")
        self.qiNU = self.Q(Z, "Proton_charge")

        self.mi = self.miNU.to("kilogram")
        self.qi = self.qiNU.to("Coulomb")

        # parse muB depending on its dimensionality
        logger.debug(f"\tGiven muB units: {muB.units:~}")
        if muB.dimensionality == self.Q("Magnetic_moment").dimensionality:
            self.mu = muB.to("keV/Tesla")
        elif muB.dimensionality == self.Q("keV").dimensionality:
            self.muB = muB.to("keV")

        # Keep the angles' value only
        self.theta0 = float(theta0)
        self.zeta0 = float(zeta0)

        self.psi0 = psi0.to("Magnetic_flux")
        self.Pzeta0 = Pzeta0.to("Canonical_momentum")
        self.t_eval = t_eval.to("seconds")

        # Corresponding NU values
        self.psi0NU = self.psi0.to("NUMagnetic_flux")
        self.Pzeta0NU = Pzeta0.to("NUCanonical_momentum")
        self.t_evalNU = self.t_eval.to("NUseconds")

        self._full_set_calculated = False
        logger.info("--> InitialConditions 1st step Initialization Complete")

    def _calculate_full_set(self, tokamak: Tokamak):
        r"""Calculates the rest of the initial conditions, which require a
        tokamak to be defined.

        These Quantities include muB/mu, rho0, psip0, Ptheta0 and the Energy,
        as well as their NU counterparts.
        """
        logger.info("==> Initializing 2nd step InitialConditions...")

        # Initial (B,i,g) and Phi values
        # Note: bigNU() and PhiNU() input and output are in [NU]
        # Symbolize the magnitude of the repspective variable with a _
        _psi0NU = self.psi0NU.m
        _theta0 = self.theta0
        _b0NU, _i0NU, _g0NU = tokamak.bfield.bigNU(_psi0NU, _theta0)
        _Phi0NU = float(tokamak.efield.PhiNU(_psi0NU, _theta0))
        B_init = self.Q(_b0NU, "NUTesla").to("Tesla")  # UNSURE:

        # Calculate the 2 other mu/muB Quantities
        if hasattr(self, "mu"):
            self.muB = (self.mu * B_init).to("keV")
        elif hasattr(self, "muB"):
            self.mu = (self.muB / B_init).to("Magnetic_moment")

        # Calculate the rest of the initial conditions.
        self.psip0 = self.Q(  # psipNU only accepts [NU]
            tokamak.qfactor.psipNU(_psi0NU),
            "NUMagnetic_flux",
        ).to("Magnetic_flux")
        self.psip0NU = self.psip0.to("NUMagnetic_flux")

        # Use magnitudes here to avoid multiplying psip0 with the proton charge
        # Note: [rho] = Magnetic_flux / Plasma_current = meters)
        _rho0NU = (self.Pzeta0NU.m + self.psip0NU.m) / _g0NU
        self.rho0NU = self.Q(_rho0NU, "NUmeters")
        self.rho0 = self.rho0NU.to("meters")

        _Ptheta0 = self.psi0NU.m + _rho0NU * _i0NU
        self.Ptheta0NU = self.Q(_Ptheta0, "NUCanonical_momentum")
        self.Ptheta0 = self.Ptheta0NU.to("Canonical_momentum")

        # Corresponding NU attributes
        self.muNU = self.mu.to("NUMagnetic_moment")
        self.muBNU = self.muB.to("NUJoule")

        # Energies
        # _ENU = _rho0NU**2 * _b0NU**2 + self.mu.m * _b0NU + _Phi0NU
        _ENU = (1 / 2) * _rho0NU**2 * _b0NU**2 + self.muNU.m * _b0NU + _Phi0NU

        self.ENU = self.Q(_ENU, "NUJoule")
        self.E = self.ENU.to("Joule")
        self.EkeV = self.E.to("keV")

        self._full_set_calculated = True
        logger.info("--> InitialConditions 2nd step Initialization Complete")

    def __repr__(self):
        t0 = self.t_eval[0].m
        dt = self.t_eval[1].m - t0
        tf = self.t_eval[-1].m
        muB = f"{self.muB:.4g~P}" if self._full_set_calculated else None

        return (
            "InitialConditions: "
            f"theta0 = {self.theta0:.4g}, "
            f"zeta0 = {self.zeta0:.4g}, "
            f"psi0 = {self.psi0:.4g~P}, "
            f"t_eval: [t0, tf, dt]=[{t0:.4g}, {tf:.4g}, {dt:.4g}]s, "
            f"muB = {muB}"
        )

    def __str__(self):

        if not self._full_set_calculated:
            return self.__repr__()

        t0 = self.t_eval[0].m
        dt = self.t_eval[1].m - t0
        tf = self.t_eval[-1].m

        return (
            colored("\nInitial Conditions :\n", "green")
            + f"{"Particle species":>23} : "
            f"{colored(self.species_name, "light_blue"):<16}\n"
            f"{"theta0":>23} : {f'{self.theta0:.4g} radians':<16} \n"
            f"{"zeta0":>23} : {f'{self.zeta0:.4g} radians':<16} \n"
            f"{"psi0":>23} : {f'{self.psi0:.4g~}':<16} ({self.psi0NU:.4g})\n"
            f"{"Pzeta0":>23} : {f'{self.Pzeta0:.4g~}':<16} "
            f"({self.Pzeta0NU:.4g~})\n"
            f"{"Energy":>23} : {f'{self.EkeV:.4g~}':<16} "
            f"({self.ENU:4g~})\n"
            f"{"Magnetic Moment":>23} : {f'{self.mu:.4g~}':<16} "
            f"({self.muNU:4g~})\n"
            f"{"muB(initial)":>23} : {f'{self.muB:.4g~}':<16} "
            f"({self.muBNU:4g~})\n"
            f"{"t_eval":>23} : "
            f"{f'[t0, tf, dt] = [{t0:.4g}, {tf:.4g}, {dt:.4g}]s':<16}\n"
        )
