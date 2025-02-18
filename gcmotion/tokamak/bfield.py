r"""
Defines the MagneticFieldBase class and all available bfield
configurations.
"""

import os
import pint
import numpy as np
import xarray as xr
from termcolor import colored
from scipy.interpolate import UnivariateSpline, RectBivariateSpline
from pint import UndefinedUnitError
from time import time

from gcmotion.utils.logger_setup import logger

from math import cos, sin, sqrt
from abc import ABC, abstractmethod

# Quantity alias for type annotations
type Quantity = pint.UnitRegistry.Quantity


class MagneticField(ABC):
    r"""Magnetic field base class"""

    def __init__(self):
        r"""Contains all the needed parameters

        The parameters should be defined as Quantities with SI units.
        """

    @abstractmethod
    def bigNU(
        self, phi: float | np.ndarray, theta: float | np.ndarray
    ) -> float | np.ndarray:
        r"""Calculates :math:`B(\psi, \theta), I(\psi, \theta), g(\psi,\
        \theta)`. Input and output must be both floats or np.ndarrays, in
        [NU].

        Used in energy contour plots.

        Parameters
        ----------
        psi : float | np.ndarray
            The :math:`\psi` value(s) in [NU].
        theta : float | np.ndarray
            The :math:`\theta` value(s).

        Returns
        -------
        float | np.ndarray
            The Calculated :math:`B(\psi, \theta), I(\psi, \theta), g(\psi,\
            \theta)` value(s) in [NU].

        """

    @abstractmethod
    def solverbNU(
        self, psi: float, theta: float
    ) -> tuple[float, float, float]:
        r"""Calculates all the values needed by the solver:
        :math:`B,I,g` (by calling ``self.bigNU()``) and the derivatives
        :math:`\dfrac{\partial B}{\partial \psi}, \dfrac{\partial B}{\partial\
        \theta}\ \dfrac{\partial I}{\partial \psi}` and :math:`\dfrac{\partial\
        g}{\partial \psi}`.

        Input and output must be floats, in [NU].

        Used inside the solver.

        .. warning::
            The derivatives are calculated with respect to :math:`\psi`, and
            **not** :math:`\psi_p`, which appear in the differential equations.
            This is accounted for inside the solver by multiplying by
            :math:`q(\psi)`, since :math:`\dfrac{\partial f}{\partial \psi_p}\
            =\dfrac{\partial f}{\partial \psi}\ \dfrac{\partial \psi}{\partial\
            \psi_p} = q\dfrac{\partial f}{\partial \psi}`

        Parameters
        ----------
        psi : float
            Value of :math:`\psi` in [NU].
        theta : float
            Value of :math:`\theta` in [NU].

        Returns
        -------
        3-tuple of floats
            Calculated values and derivatives in [NU].
        """


class NumericalMagneticField(MagneticField):
    r"""Numerical Magnetic field base class

    Opens the dataset and creates the splines needed for the querry methods.
    Also defines Bmin and Bmax, used in the parabolas.

    ``solverB`` and ``bigNU`` work identically to the analytic Magnetic field
    methods.

    Parameters
    ----------
    filename : str
        The "\*.nc" file located at gcmotion/tokamak/reconstructed.

    Attributes
    ----------
    Bmin, Bmax : Quantities
        The minimum and maximum magnetic field values.
    theta_min, theta_max : Quantities
        The :math:`\theta` coordinates where the magnetic field takes its
        minimum/maximum value.
    psi_min, psi_max : Quantities
        The :math:`\psi` coordinates where the magnetic field takes its
        minimum/maximum value.

    """

    def __init__(self, filename: str):

        # Open the dataset
        parent = os.path.dirname(__file__)  # Relative to this directory
        path = os.path.join(parent, "reconstructed", filename)
        try:
            dataset = xr.open_dataset(path)
            self.dataset = dataset
            logger.info("Dataset initialized correctly.")
        except FileNotFoundError:
            logger.error("Error opening Dataset.")
            raise FileNotFoundError(f"No file found at '{path}'")

        # Extract the arrays
        psi_values = dataset.psi.data
        theta_values = dataset.boozer_theta.data
        b_values = dataset.b_field_norm.data.T
        i_values = dataset.I_norm.data
        g_values = dataset.g_norm.data

        # Create splines
        self.b_spline = RectBivariateSpline(
            x=theta_values,
            y=psi_values,
            z=b_values,
        )
        # NOTE: Do not use the dataset bfield derivatives. They introduce small
        # non-Hamiltonian terms, resulting in a noticeable fluxuation and even
        # loss of Energy, due to error propagation through 2 different splines.
        # Use the derivatives calculated from the bfield spline instead.
        self.db_dpsi_spline = self.b_spline.partial_derivative(
            dx=0,
            dy=1,
        )
        self.db_dtheta_spline = self.b_spline.partial_derivative(
            dx=1,
            dy=0,
        )
        self.i_spline = UnivariateSpline(
            x=psi_values,
            y=i_values,
        )
        self.g_spline = UnivariateSpline(
            x=psi_values,
            y=g_values,
        )

        self.ider_spline = self.i_spline.derivative(n=1)
        self.gder_spline = self.g_spline.derivative(n=1)

        # Useful attributes
        Q = pint.UnitRegistry.Quantity
        _B0 = float(dataset.Baxis.data)  # Tesla
        self.B0 = Q(_B0, "Tesla")

        # Magnetic field strength extremum
        start = time()
        da = dataset.b_field_norm
        mins = da.where(da == da.min(), drop=True).squeeze()
        maxs = da.where(da == da.max(), drop=True).squeeze()
        self.Bmin = Q(mins.values, "NUTesla").to("Tesla")
        self.Bmax = Q(maxs.values, "NUTesla").to("Tesla")
        self.theta_min = float(mins.boozer_theta.values.flatten()[0])
        self.theta_max = float(maxs.boozer_theta.values.flatten()[0])

        # WARN: NU units must have been defined in the unit registry already.
        # The quantity_constructor module sets the global "application
        # registry" when imported, and *then* defines NU units. This means
        # quantifying psi raises an exception if the object is created without
        # having the Quantity constructor instantiated first. There is no way
        # around that, but there is really no reason to do that.
        # WARN: "type(self.Bmax) is Q" returns False which might cause
        # problems.
        try:
            self.psi_min = Q(mins.psi.values, "NUMagnetic_flux")
            self.psi_max = Q(maxs.psi.values, "NUMagnetic_flux")
        except UndefinedUnitError:
            logger.warning(
                "psi coordinates of Bmin and Bmax cannot be defined. "
                "Ensure that the Quantity Constructor has been instantiated "
                "correctly first."
            )
        end = time()
        duration = Q(end - start, "seconds")
        logger.info(f"Numerical bfield extremum search took {duration:.4g~#P}")

        self.is_numerical = True

    def bigNU(self, psi: float | np.ndarray, theta: float | np.ndarray):
        theta = theta % (2 * np.pi)
        b = self.b_spline(x=theta, y=psi, grid=False)
        i = self.i_spline(x=psi)
        g = self.g_spline(x=psi)

        return (b, i, g)

    def solverbNU(self, psi: float, theta: float):

        theta = theta % (2 * np.pi)
        # Field and currents
        b, i, g = self.bigNU(psi, theta)

        # Field derivatives
        db_dpsi = self.db_dpsi_spline(x=theta, y=psi, grid=False)
        db_dtheta = self.db_dtheta_spline(x=theta, y=psi, grid=False)

        # Current derivatives
        i_der = self.ider_spline(psi)
        g_der = self.gder_spline(psi)

        # Pack them up
        currents = (i, g)
        b_der = (db_dpsi, db_dtheta)
        currents_der = (i_der, g_der)

        return b, b_der, currents, currents_der


# ============================================================================


class LAR(MagneticField):
    r"""Initializes the standard Large Aspect Ratio magnetic field.

    Parameters
    ----------
    B0 : Quantity
        The strength of the magnetic field *on the magnetic axis*
        in [T].
    i : Quantity
        The toroidal current in units of [Plasma Current].
    g : Quantity
        The poloidal current in units of [Plasma Current].

    """

    def __init__(self, B0: Quantity, i: Quantity, g: Quantity):

        # SI Quantities
        self.B0 = B0.to("Tesla")
        self.i = i.to("Plasma_Current")
        self.g = g.to("Plasma_Current")

        # [NU] Conversions
        self.B0NU = B0.to("NUTesla")
        self.iNU = i.to("NUPlasma_Current")
        self.gNU = g.to("NUPlasma_Current")

        # Unitless quantities, makes it a bit faster if defined here
        for key, value in self.__dict__.copy().items():
            self.__setattr__("_" + key, value.magnitude)

        # Flags
        self.has_i = bool(i.m)
        self.plain_name = "LAR"
        self.is_analytic = True

        # Minimum/Maximum values and locations
        Q = pint.UnitRegistry.Quantity
        try:
            self.psi_wall = Q(1, "psi_wall")
            self.theta_min, self.theta_max = 0, np.pi
            self.psi_min = self.psi_max = self.psi_wall.to("NUMagnetic_flux").m

            _BminNU = self.bigNU(psi=self.psi_min, theta=self.theta_min)[0]
            _BmaxNU = self.bigNU(psi=self.psi_max, theta=self.theta_max)[0]
            self.Bmin = Q(_BminNU, "NUTesla").to("Tesla")
            self.Bmax = Q(_BmaxNU, "NUTesla").to("Tesla")

        except UndefinedUnitError:
            logger.warning(
                "psi coordinates of Bmin and Bmax cannot be defined. "
                "Ensure that the Quantity Constructor has been instantiated "
                "correctly first."
            )

    def bigNU(self, psi: float | np.ndarray, theta: float | np.ndarray):

        if isinstance(psi, (int, float)):
            b = 1 - sqrt(2 * psi) * cos(theta)
            g = self._gNU
            i = self._iNU
        elif isinstance(psi, np.ndarray):
            b = 1 - np.sqrt(2 * psi) * np.cos(theta)
            g = np.tile(self._gNU, psi.shape)
            i = np.tile(self._iNU, psi.shape)

        return (b, i, g)

    def solverbNU(self, psi: float, theta: float):

        # Field and currents
        b, i, g = self.bigNU(psi, theta)

        # Field derivatives
        root = sqrt(2 * psi)
        cos_theta = cos(theta)
        b_der_psi = -cos_theta / root
        b_der_theta = root * sin(theta)

        # Current derivatives
        i_der = 0
        g_der = 0

        # Pack them up
        currents = (i, g)
        b_der = (b_der_psi, b_der_theta)
        currents_der = (i_der, g_der)

        return b, b_der, currents, currents_der

    def __repr__(self):
        return (
            colored("LAR", "light_blue")
            + f": B0={self.B0:.4g~}, I={self.i:.4g~}, g={self.g:.4g~}."
        )


class SmartPositive(NumericalMagneticField):
    r"""Initializes a bfield object with numerical data from the Smart Tokamak
    with **Positive** Triangularity.

    The dataset must be stored in
    *./gcmotion/tokamak/reconstructed/smart_positive.nc*.

    """

    def __init__(self):
        filename = "smart_positive.nc"
        super().__init__(filename=filename)

        self.plain_name = "Smart - Positive"

    def __repr__(self):
        return (
            colored("Smart - Positive", "light_blue") + f": B0={self.B0:.4g~}."
        )


class SmartNegative(NumericalMagneticField):
    r"""Initializes a bfield object with numerical data from the Smart Tokamak
    with **Negative** Triangularity.

    The dataset must be stored in
    *./gcmotion/tokamak/reconstructed/smart_negative.nc*.

    """

    def __init__(self):
        filename = "smart_negative.nc"
        super().__init__(filename=filename)

        self.plain_name = "Smart - Negative"

    def __repr__(self):
        return (
            colored("Smart - Negative", "light_blue") + f": B0={self.B0:.4g~}."
        )


class DivertorPositive(NumericalMagneticField):
    r"""Initializes a bfield object with numerical data from the Divertor
    Tokamak with **Positive** Triangularity.

    The dataset must be stored in
    *./gcmotion/tokamak/reconstructed/divertor_positive.nc*.

    """

    def __init__(self):
        filename = "divertor_negative.nc"
        super().__init__(filename=filename)

        self.plain_name = "Divertor - Positive"

    def __repr__(self):
        return (
            colored("Divertor - Positive", "light_blue")
            + f": B0={self.B0:.4g~}."
        )


class DivertorNegative(NumericalMagneticField):
    r"""Initializes a bfield object with numerical data from the Divertor
    Tokamak with **Negative** Triangularity.

    The dataset must be stored in
    *./gcmotion/tokamak/reconstructed/divertor_negative.nc*.

    """

    def __init__(self):
        filename = "divertor_negative.nc"
        super().__init__(filename=filename)

        self.plain_name = "Divertor - Negative"

    def __repr__(self):
        return (
            colored("Divertor - Negative", "light_blue")
            + f": B0={self.B0:.4g~}."
        )
