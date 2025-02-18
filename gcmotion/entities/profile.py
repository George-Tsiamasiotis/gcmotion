r"""
==============
Profile Entity
==============

A Profile entity defines the Constants of motion E, Pzeta and mu, as well as
the particle species. Its methods findEnergy, findPzeta and findmu ignore the
respective constant and return a new Quantity for a given pair of psi, theta.
"""

import pint
import numpy as np
from termcolor import colored
from typing import Literal

from gcmotion.utils.logger_setup import logger

from gcmotion.entities.tokamak import Tokamak
from gcmotion.configuration.physical_constants import PhysicalConstants

type Quantity = pint.Quantity
type SupportedSpecies = Literal["p", "e", "D", "T", "He3", "He4"]


class Profile:
    r"""A Profile entity describes an unperturbed equilibrium with a given
    Tokamak configuration and with at least 2/3 fixed Constants of Motion.

    By fixing 2 COMs and letting the other one be variable, we can perform many
    useful analyses.

    Note that even if specified, the 3rd COM will be ignored in an analysis
    that by its nature lets it vary. For example, contouring over Energy
    levels, will simply ignore the E parameter.

    Parameters
    ----------
    tokamak : :py:class:`~gcmotion.Tokamak`
        The Tokamak entity.
    species : {'p', 'e', 'D', 'T', 'He3', 'He4'}
        The particle's species. This field is case-insensitive.
    mu : Quantity
        The Magnetic Moment COM :math:`\mu`.
    Pzeta : Quantity
        The Canonical Momentum COM :math:`P_\zeta`.
    E : Quantity
        The E COM :math:`E`.

    Attributes
    ----------
    species: str
        The profile's particle species
    mi, qi, miNU, qiNU : Quantities
        The profile's particle mass and charge in SI/NU.
    mu, muNU: Quantities
        The magnetic moment :math:`\mu` in SI/NU. Must have dimensionality of
        *[current]x[area]*.
    Pzeta, PzetaNU: Quantities
        The :math:`P_\zeta` canonical momentum in SI/NU. Must have
        dimensionality of *magnetic flux*.
    E, ENU: Quantities
        The Energy constant of motion in SI/NU.

    Examples
    --------
    How to create a `Profile` object.

    >>> import gcmotion as gcm
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
    >>> i = Q(0, "NUPlasma_current")
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
    >>> # Create a Profile
    >>> profile = gcm.Profile(
    ...     tokamak=tokamak,
    ...     species=species,
    ...     mu=Q(1e-5, "NUMagnetic_moment"),
    ...     Pzeta=Q(-0.015, "NUCanonical_momentum")
    ... )

    """

    def __init__(
        self,
        tokamak: Tokamak,
        species: SupportedSpecies,
        mu: Quantity = None,
        Pzeta: Quantity = None,
        E: Quantity = None,
    ):

        logger.info("==> Initializing Profile...")

        E_str = None if E is None else f"{E:.4g~}"
        mu_str = None if mu is None else f"{mu:.4g~}"
        Pzeta_str = None if Pzeta is None else f"{Pzeta:.4g~}"
        logger.debug(f"\tParameters: " f"mu={mu_str}, {Pzeta_str}, {E_str}")

        # Grab attributes from tokamak object
        self.tokamak = tokamak
        self.__dict__.update(self.tokamak.__dict__)
        logger.debug("\tCopied tokamak's attributes to self.")

        # Define species
        self.Q = tokamak.Q
        self.species = species.lower()
        self.species_name = getattr(
            PhysicalConstants, self.species + "_name", None
        )

        # Grab particle's mass and charge in NU
        M = getattr(PhysicalConstants, self.species + "_M")
        Z = getattr(PhysicalConstants, self.species + "_Z")

        self.miNU = self.Q(M, "Proton_mass")
        self.qiNU = self.Q(Z, "Proton_charge")

        self.mi = self.miNU.to("kilogram")
        self.qi = self.qiNU.to("Coulomb")

        # Constants of motion
        if mu is None:
            self._mu = self._muNU = None
        else:
            self._mu = mu.to("Magnetic_moment")
            self._muNU = self._mu.to("NUMagnetic_moment")
            logger.debug(f"\tSet {self._mu=:.4g~} and {self._muNU=:.4g~}")

        if Pzeta is None:
            self._Pzeta = self._PzetaNU = None
        else:
            self._Pzeta = Pzeta.to("Canonical_momentum")
            self._PzetaNU = self._Pzeta.to("NUCanonical_momentum")
            logger.debug(
                f"\tSet {self._Pzeta=:.4g~} and {self._PzetaNU=:.4g~}"
            )

        if E is None:
            self._E = self._ENU = None
            logger.debug(f"\tSet {self.E=} and {self.ENU=}")
        else:
            self._E = E.to("keV")
            self._ENU = self._E.to("NUJoule")
            logger.debug(f"\tSet {self._E=:.4g~} and {self._ENU=:.4g~}")

        logger.info("--> Profile Initialization Complete.")

    # NOTE: Unfortunately those are need to update correctly both SI and NU
    # Quantities

    @property
    def mu(self):
        return self._mu

    @property
    def muNU(self):
        return self._muNU

    @mu.setter
    def mu(self, new_mu: Quantity):
        self._mu = new_mu.to("Magnetic_moment")
        self._muNU = self._mu.to("NUmagnetic_moment")

    @muNU.setter
    def muNU(self, new_mu):
        self.mu = new_mu

    @property
    def Pzeta(self):
        return self._Pzeta

    @property
    def PzetaNU(self):
        return self._PzetaNU

    @Pzeta.setter
    def Pzeta(self, new_Pzeta: Quantity):
        self._Pzeta = new_Pzeta.to("Canonical_momentum")
        self._PzetaNU = self._Pzeta.to("NUCanonical_momentum")

    @PzetaNU.setter
    def PzetaNU(self, new_Pzeta):
        self.Pzeta = new_Pzeta

    @property
    def E(self):
        return self._E

    @property
    def ENU(self):
        return self._ENU

    @E.setter
    def E(self, new_E: Quantity):
        self._E = new_E.to("keV")
        self._ENU = self._E.to("NUJoule")

    @ENU.setter
    def ENU(self, new_E):
        self.E = new_E

    def findPtheta(self, psi: Quantity, units: str):
        r"""Calculates Ptheta from psi. "Pzeta"" must be defined.

        Output units are the same as input units.

        Only applicable in the absence of perturbations.

        Parameters
        ----------
        psi : Quantity
            The psi Quantity.
        units : str
            The Ptheta units. Must have dimensionality of canonical momentum
            (orbital momentum \ [energy] * [time]),

        Returns
        -------
        Quantity
            The calculated Ptheta Quantity.

        """
        # Do all operations in NU units, and Quantify at the end.
        psiNU = psi.to("NUMagnetic_flux")
        psiNU = psiNU.magnitude

        # Calculate currents. B not needed so theta value doesnt matter
        _, iNU, gNU = self.bfield.bigNU(psiNU, 0)
        psipNU = self.qfactor.psipNU(psiNU)

        rhoNU = (self.PzetaNU.m + psipNU) / gNU
        PthetaNU = psiNU + rhoNU * iNU

        # Quantify, convert to input units and return
        PthetaNU = self.Q(PthetaNU, "NUCanonical_momentum")
        return PthetaNU.to(units)

    def findEnergy(
        self, psi: Quantity, theta: float, units: str, potential: bool = True
    ):
        r"""Calculates the Energy of a particle characterized by a (psi, theta)
        pair. Both `Pzeta` and `mu` must be defined.

        Parameters
        ----------
        psi : Quantity
            The particle's psi Quantity.
        theta : float
            The particle's :math:`\theta` angle.
        units : str
            The returned Energy units.
        potential : bool, optional
            Whether or not to add the electric potential term in the energy.
            Defaults to True.

        Returns
        -------
        Quantity
            The calculated Energy Quantity in the specified units.
        """
        # Do all operations in NU floats, and Quantify at the end.
        _ = psi.to("NUMagnetic_flux")
        psiNU = _.magnitude

        # Calculate currents. B not needed so theta value doesnt matter
        bNU, iNU, gNU = self.bfield.bigNU(psiNU, theta)
        psipNU = self.qfactor.psipNU(psiNU)

        rhoNU = (self.PzetaNU.m + psipNU) / gNU

        EnergyNU = (
            1 / 2
        ) * rhoNU**2 * bNU**2 + self.muNU.m * bNU  # Without potential

        if potential:
            PhiNU = self.efield.PhiNU(psiNU, theta)
            EnergyNU += PhiNU

        # Quantify, convert to input units and return
        EnergyNU = self.Q(EnergyNU, "NUJoule")
        return EnergyNU.to(units)

    def findPzeta(
        self, psi: Quantity, theta: float, units: str, potential: bool = True
    ):
        r"""Calculates the "Pzeta" of a particle characterized by a (psi,
        theta) pair. Both `Energy` and `mu` must be defined.

        Parameters
        ----------
        psi : Quantity
            The particle's psi Quantity.
        theta : float | np.ndarray
            The particle's :math:`\theta` angle.
        units : str
            The returned Pzeta units.
        potential : bool, optional
            Whether or not to add the electric potential term in the
            calculation. Defaults to True.

        Returns
        -------
        Quantity
            The calculated Pzeta Quantity in the specified units.

        """
        # Do all operations in NU floats, and Quantify at the end.
        psiNU = psi.to("NUMagnetic_flux")
        psiNU = psiNU.magnitude

        # Calculate currents. B not needed so theta value doesnt matter
        bNU, iNU, gNU = self.bfield.bigNU(psiNU, theta)
        psipNU = self.qfactor.psipNU(psiNU)
        psipNU = self.qfactor.psipNU(psiNU)

        if potential:
            PhiNU = self.efield.PhiNU(psiNU, theta)
        else:
            PhiNU = np.zeros(psiNU.shape)  # Keep psi's shape

        PzetaNU = (
            (2 * gNU**2 / bNU**2) * (self.ENU.m - self.muNU.m * bNU - PhiNU)
        ) ** (1 / 2) - psipNU

        # Quantify, convert to input units and return
        PzetaNU = self.Q(PzetaNU, "NUCanonical_momentum")
        return PzetaNU.to(units)

    def findmu(
        self, psi: Quantity, theta: float, units: str, potential: bool = True
    ):
        r"""Calculates the "mu " of a particle characterized by a (psi, theta)
        pair. Both `Energy` and `Pzeta` must be defined.

        Parameters
        ----------
        psi : Quantity
            The particle's psi Quantity.
        theta : float
            The particle's :math:`\theta` angle.
        units : str
            The returned mu units.
        potential : bool, optional
            Whether or not to add the electric potential term in the
            calculation. Defaults to True.

        Returns
        -------
        Quantity
            The calculated mu Quantity in the specified units.
        """
        # Do all operations in NU floats, and Quantify at the end.
        psiNU = psi.to("NUMagnetic_flux")
        psiNU = psiNU.magnitude

        # Calculate currents. B not needed so theta value doesnt matter
        bNU, iNU, gNU = self.bfield.bigNU(psiNU, theta)
        psipNU = self.qfactor.psipNU(psiNU)
        psipNU = self.qfactor.psipNU(psiNU)

        if potential:
            PhiNU = self.efield.PhiNU(psiNU, theta)
        else:
            PhiNU = 0 * psi  # Keep psi's shape

        rhoNU = (self.PzetaNU.m + psipNU) / gNU

        # UNSURE: Can μ be negative?
        muNU = (self.ENU.m - PhiNU) / bNU - rhoNU**2 * bNU

        # Quantify, convert to input units and return
        muNU = self.Q(muNU, "NUMagnetic_moment")
        return muNU.to(units)

    def _rhosign(self, psi: np.ndarray) -> np.ndarray:
        r"""Calculates the sign of rho from a given psi[NU].

        Needed to classify an orbit as co- or counter-passing. Makes no sense
        to be used in trapped orbits.

        Parameters
        ----------
        psi : np.ndarray
            The psi values in NU

        Returns
        -------
        bool
            True if all values are positive(co), else False(counter).
        """
        psip = self.tokamak.qfactor.psipNU(psi)
        # UNSURE: no need for g since its positive
        return bool(np.all(self.PzetaNU.m + psip > 0))

    def _findPzeta(
        self,
        psi: Quantity,
        theta: float,
        energy,
        units: str,
        potential: bool = True,
    ):
        # Do all operations in NU floats, and Quantify at the end.
        psiNU = psi.to("NUMagnetic_flux")
        psiNU = psiNU.magnitude
        energyNU = energy.to("NUJoule")
        energyNU = energy.magnitude

        # Calculate currents. B not needed so theta value doesnt matter
        bNU, iNU, gNU = self.bfield.bigNU(psiNU, theta)
        psipNU = self.qfactor.psipNU(psiNU)
        psipNU = self.qfactor.psipNU(psiNU)

        if potential:
            PhiNU = self.efield.PhiNU(psiNU, theta)
        else:
            PhiNU = np.zeros(psiNU.shape)  # Keep psi's shape

        PzetaNU = (
            (2 * gNU**2 / bNU**2) * (energyNU - self.muNU.m * bNU - PhiNU)
        ) ** (1 / 2) - psipNU

        # Quantify, convert to input units and return
        PzetaNU = self.Q(PzetaNU, "NUCanonical_momentum")
        return PzetaNU.to(units)

    def __repr__(self):
        string = Tokamak.__repr__(self)
        string += (
            "PhysicalParameters: "
            f"species = {self.species}, "
            f"mu = {self.mu}, "
            f"Pzeta = {self.Pzeta}, "
            f"E = {self.E}\n"
        )
        return string

    def __str__(self):
        mu = "None" if self.mu is None else f"{self.mu:.4g~}"
        muNU = "None" if self.mu is None else f"{self.muNU:.4g~}"
        Pzeta = "None" if self.Pzeta is None else f"{self.Pzeta:.4g~}"
        PzetaNU = "None" if self.Pzeta is None else f"{self.PzetaNU:.4g~}"
        E = "None" if self.E is None else f"{self.E:.4g~}"
        ENU = "None" if self.ENU is None else f"{self.ENU:.4g~}"

        string = Tokamak.__str__(self)
        string += colored("\nConstans of Motion\n", "green")
        string += (
            f"{"Particle species":>23} : "
            f"{colored(self.species_name, "light_blue"):<16}\n"
            f"{"mu":>23} : {mu:<16}({muNU})\n"
            f"{"Pzeta":>23} : {f'{Pzeta}':<16}({PzetaNU})\n"
            f"{"E":>23} : {f'{E}':<16}({ENU})\n"
        )

        return string
