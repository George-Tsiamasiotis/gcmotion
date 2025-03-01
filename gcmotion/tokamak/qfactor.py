r"""
Defines the QFactor Base class and all available q-factor configurations.
"""

import os
import pint
import numpy as np
import xarray as xr


from abc import ABC, abstractmethod
from termcolor import colored
from math import sqrt, atan, asinh
from scipy.special import hyp2f1
from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline

from gcmotion.utils.logger_setup import logger
from gcmotion.utils.precompute import precompute_hyp2f1


# Quantity alias for type annotations
type Quantity = pint.Quantity


class QFactor(ABC):
    r"""q Factor base class"""

    @abstractmethod
    def __init__(self):
        r"""Contains all the needed parameters."""

    @abstractmethod
    def solverqNU(self, psi: float | np.ndarray) -> float | np.ndarray:
        r"""Calculates :math:`q(\psi)`.
        Input and output are both floats or np.ndarrays, in [NU].

        Used inside the solver.

        Parameters
        ----------
        psi : float | np.ndarray
            Value(s) of :math:`\psi` in NU.

        Returns
        -------
        float | np.ndarray
            Calculated :math:`\psi_p(\psi)` in NU.

        """

    @abstractmethod
    def psipNU(self, psi: float | np.ndarray) -> float | np.ndarray:
        r"""Calculates :math:`\psi_p(\psi)`.
        Input and output are both floats or np.ndarrays, in [NU].

        Parameters
        ----------
        psi : float | np.ndarray
            Value(s) of :math:`\psi` in NU.

        Returns
        -------
        float | np.ndarray
            Calculated :math:`\psi_p(\psi)` in NU.

        """


class NumericalQFactor(QFactor):
    r"""Numerical QFactor base class.

    Opens the dataset and creates the splines needed for the querry methods.

    ``solverq`` and ``psipNU`` work identically to the analytic QFactor
    methods.

    Parameters
    ----------
    filename : str
        The "\*.nc" file located at gcmotion/tokamak/reconstructed.

    """

    is_analytical = False
    is_numerical = True

    def __init__(self, filename: str):
        # Open the dataset
        parent = os.path.dirname(__file__)
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
        q_values = dataset.q.data

        # Extrapolate psi to containn psi=0
        psi_values = np.insert(psi_values, 0, 0)

        # Extrapolate q to contain the value q(psi=0), which is the same as
        # q[1], to avoid errors when integrating.
        q_values = np.insert(q_values, 0, q_values[0])

        # Create q spline
        self.qspline = UnivariateSpline(x=psi_values, y=q_values)

        # Create psip spline
        iota_values = 1 / q_values
        iota_spline = UnivariateSpline(x=psi_values, y=iota_values)
        self.psip_spline = iota_spline.antiderivative(n=1)

        # Calculate useful attributes
        self.q0 = q_values[0]
        self.q_wall = q_values[-1]

    def solverqNU(self, psi: float) -> float:
        if isinstance(psi, float):
            return float(self.qspline(psi))
        elif isinstance(psi, np.ndarray):
            return self.qspline(psi)

    def psipNU(self, psi: float | np.ndarray) -> float | np.ndarray:
        if isinstance(psi, float):
            return float(self.psip_spline(psi))
        elif isinstance(psi, np.ndarray):
            return self.psip_spline(psi)


# ============================================================================


class Unity(QFactor):
    r"""Initializes an object q with :math:`q(\psi) = 1`
    and :math:`\psi_p=\psi`."""

    is_analytical = True
    is_numerical = False

    def __init__(self):
        pass

    def solverqNU(self, psi: float | np.ndarray) -> float | np.ndarray:
        r"""Always returns 1."""
        return psi / psi

    def psipNU(self, psi):
        """Always returns `psi`."""
        return psi

    def __repr__(self):
        return colored("Unity", "light_blue")


class Parabolic(QFactor):
    r"""Initializes an object q with

    .. math::

        q(\psi) = q_0 + (q_{wall}-q_0)\bigg(\dfrac{\psi}{\psi_{wall}}\
        \bigg)^2

    :math:`\psi_p(\psi)` is calculated from:

    .. math::

        \psi_p(\psi) = \dfrac{\psi_{wall}}{\sqrt{q_0 (q_{wall}-q_0)}}\
        \arctan\bigg( \dfrac{\psi\sqrt{q_{wall}-q_0}}{\psi_{wall}\sqrt{q_0}}\
        \bigg)

    Parameters
    ----------
    a : Quantity
        float The tokamak's minor radius in [m].
    B0 : Quantity
        The Magnetic field's strength in [T]
    q0 : float
        q-value at the magnetic axis.
    q_wall : float
        q_value at the wall.

    """

    is_analytical = True
    is_numerical = False

    def __init__(
        self,
        a: Quantity,
        B0: Quantity,
        q0: float,
        q_wall: float,
    ):

        # SI Quantities
        self.B0 = B0.to("Tesla")
        self.a = a.to("meters")
        self.psi_wall = (B0 * a**2 / 2).to("Magnetic_flux")
        self.psi_wallNU = self.psi_wall.to("NUMagnetic_flux")

        # Unitless quantities, makes it a bit faster if defined here
        for key, value in self.__dict__.copy().items():
            self.__setattr__("_" + key, value.magnitude)

        self.sra = sqrt(q0)
        self.srb = sqrt(q_wall - q0)

        # Purely Numerical Parameters
        self.q0 = q0
        self.q_wall = q_wall

    def solverqNU(self, psi):
        return (
            self.q0 + (self.q_wall - self.q0) * (psi / self._psi_wallNU) ** 2
        )

    def psipNU(self, psi):
        if isinstance(psi, float):
            return (self._psi_wallNU / (self.sra * self.srb)) * atan(
                self.srb * psi / (self.sra * self._psi_wallNU)
            )
        else:
            return (self._psi_wallNU / (self.sra * self.srb)) * np.atan(
                self.srb * psi / (self.sra * self._psi_wallNU)
            )

    def __repr__(self):
        return (
            colored("Parabolic", "light_blue")
            + f": q0={self.q0:.4g}, q_wall={self.q_wall:.4g}."
        )


class Hypergeometric(QFactor):
    r"""Initializes an object q with:

    .. math::

        q(\psi) = q_0\bigg\{ 1 + \bigg[ \bigg(\dfrac{q_{wall}}{q_0}\
        \bigg)^n -1 \bigg] \
        \bigg( \dfrac{\psi}{\psi_{wall}} \bigg)^n \bigg\}^{1/n}

    :math:`\psi_p(\psi)` is calculated from:

    .. math::

        \psi_p(\psi) = \dfrac{\psi}{q_0} \phantom{1}_2 F_1\
        \bigg[ \dfrac{1}{n}, \dfrac{1}{n}, 1+\dfrac{1}{n},
        \bigg(1 - \bigg( \dfrac{q_{wall}}{q_0} \bigg)^n\bigg)
        \bigg( \dfrac{\psi}{\psi_{wall}} \bigg)^n \bigg]

    where :math:`\phantom{1}_2 F_1` the hypergeometric function.

    Parameters
    ----------
    a : Quantity
        float The tokamak's minor radius in [m].
    B0 : Quantity
        The Magnetic field's strength in [T].
    q0 : float
        q-value at the magnetic axis.
    q_wall : float
        q_value at the wall.
    n : int
        Order of equillibrium (1: peaked, 2: round, 4: flat).
    """

    is_analytical = True
    is_numerical = False

    def __init__(
        self,
        a: Quantity,
        B0: Quantity,
        q0: float,
        q_wall: float,
        n: int,
    ):

        # SI Quantities
        self.B0 = B0.to("Tesla")
        self.a = a.to("meters")
        self.psi_wall = (B0 * a**2 / 2).to("Magnetic_flux")

        # [NU] Conversions
        self.psi_wallNU = self.psi_wall.to("NUMagnetic_flux")

        # Unitless quantities, makes it a bit faster if defined here
        for key, value in self.__dict__.copy().items():
            self.__setattr__("_" + key, value.magnitude)

        # Purely Numerical Quantities
        self.q0 = q0
        self.q_wall = q_wall
        self.n = n

    def solverqNU(self, psi):
        return self.q0 * (
            1
            + ((self.q_wall / self.q0) ** self.n - 1)
            * (psi / self._psi_wallNU) ** self.n
        ) ** (1 / self.n)

    def psipNU(self, psi):

        a = b = 1 / self.n
        c = 1 + 1 / self.n
        z = (1 - (self.q_wall / self.q0) ** self.n) * (
            psi / self._psi_wallNU
        ) ** self.n
        return psi / self.q0 * hyp2f1(a, b, c, z)

    def __repr__(self):
        return (
            colored("Hypergeometric", "light_blue")
            + f": q0={self.q0:.4g}, q_wall={self.q_wall:.4g}, n={self.n:.4g}."
        )


class PrecomputedHypergeometric(Hypergeometric):
    r"""Same as the Hypergeometric qfactor but with precomputed hyp2f1.

    Since the calculation of hyp2f1 is by a lot slower than any other
    calculation, it makes sense to precompute an array with all the needed
    values and make a spline on it.

    Warning
    -------
    The z-values span is set to the corresponding values for psi between 0 and
    psi_wall. If more values are needed, zspan must be adjusted accordingly,
    otherwise the spline simply extrapolates.
    """

    def __init__(
        self,
        a: Quantity,
        B0: Quantity,
        q0: float,
        q_wall: float,
        n: int,
    ):

        super().__init__(a=a, B0=B0, q0=q0, q_wall=q_wall, n=n)

        # Make z spline
        zspan = np.sort([1 - (self.q_wall / self.q0) ** self.n, 0])
        z, values = precompute_hyp2f1(n=n, zspan=zspan)
        self.z_spline = InterpolatedUnivariateSpline(x=z, y=values)
        logger.info("Using Precomputed Hyp2F1 values.")

    def psipNU(self, psi):

        z = (1 - (self.q_wall / self.q0) ** self.n) * (
            psi / self._psi_wallNU
        ) ** self.n
        hyp = self.z_spline(z)
        return psi / self.q0 * hyp

    def __repr__(self):
        return (
            colored("Hypergeometric(Precomputed)", "light_blue")
            + f": q0={self.q0:.4g}, q_wall={self.q_wall:.4g}, n={self.n:.4g}."
        )


# ============================================================================


class SmartPositive(NumericalQFactor):
    r"""Initializes an object q with numerical data from the Smart Tokamak with
    **Positive** Triangularity.

    The dataset must be stored in
    *./gcmotion/tokamak/reconstructed/smart_positive.nc*.

    """

    def __init__(self):
        filename = "smart_positive.nc"
        super().__init__(filename=filename)

    def __repr__(self):
        return (
            colored("Smart - Positive", "light_blue")
            + f": q0={self.q0:.4g}, q_wall={self.q_wall:.4g}."
        )


class SmartNegative(NumericalQFactor):
    r"""Initializes an object q with numerical data from the Smart Tokamak with
    **Negative** Triangularity.

    The dataset must be stored in
    *./gcmotion/tokamak/reconstructed/smart_negative.nc*.

    """

    def __init__(self):
        filename = "smart_negative.nc"
        super().__init__(filename=filename)

    def __repr__(self):
        return (
            colored("Smart - Negative", "light_blue")
            + f": q0={self.q0:.4g}, q_wall={self.q_wall:.4g}."
        )


class SmartNegative2(NumericalQFactor):
    r"""Initializes an object q with numerical data from the Smart Tokamak with
    **Negative** Triangularity.

    The dataset must be stored in
    *./gcmotion/tokamak/reconstructed/smart_negative2.nc*.

    """

    def __init__(self):
        filename = "smart_negative2.nc"
        super().__init__(filename=filename)

    def __repr__(self):
        return (
            colored("Smart - Negative 2", "light_blue")
            + f": q0={self.q0:.4g}, q_wall={self.q_wall:.4g}."
        )


class DTTPositive(NumericalQFactor):
    r"""Initializes an object q with numerical data from the Divertor Test
    Tokamak with **Positive** Triangularity.

    The dataset must be stored in
    *./gcmotion/tokamak/reconstructed/dtt_positive.nc*.

    """

    def __init__(self):
        filename = "dtt_positive.nc"
        super().__init__(filename=filename)

    def __repr__(self):
        return (
            colored("DTT - Negative", "light_blue")
            + f": q0={self.q0:.4g}, q_wall={self.q_wall:.4g}."
        )


class DTTNegative(NumericalQFactor):
    r"""Initializes an object q with numerical data from the Divertor Test
    Tokamak with **Negative** Triangularity.

    The dataset must be stored in
    *./gcmotion/tokamak/reconstructed/dtt_negative.nc*.

    """

    def __init__(self):
        filename = "dtt_negative.nc"
        super().__init__(filename=filename)

    def __repr__(self):
        return (
            colored("DTT - Negative", "light_blue")
            + f": q0={self.q0:.4g}, q_wall={self.q_wall:.4g}."
        )


class Chris(QFactor):
    r"""Chris's thesis qfactor."""

    def __init__(self, B0: Quantity, a: Quantity, q0: float, q_wall: float):

        # SI Quantities
        self.B0 = B0.to("Tesla")
        self.a = a.to("meters")
        self.psi_wall = (B0 * a**2 / 2).to("Magnetic_flux")

        # [NU] Conversions
        self.psi_wallNU = self.psi_wall.to("NUMagnetic_flux")

        # Unitless quantities, makes it a bit faster if defined here
        for key, value in self.__dict__.copy().items():
            self.__setattr__("_" + key, value.magnitude)

        # Purely Numerical Quantities
        self.q0 = q0
        self.q_wall = q_wall

        self.is_analytic = True

    def solverqNU(self, psi: float):

        return self.q0 * (
            1
            + (-1 + (self.q_wall / self.q0) ** 2) * (psi / self._psi_wall) ** 2
        ) ** (1 / 2)

    def psipNU(self, psi: float):
        if isinstance(psi, (int, float)):
            sinh = asinh(
                (psi * sqrt(-1 + (self.q_wall / self.q0) ** 2))
                / self._psi_wall
            )
        else:
            sinh = np.arcsinh(
                (psi * sqrt(-1 + (self.q_wall / self.q0) ** 2))
                / self._psi_wall
            )
        return (
            self._psi_wall
            / (self.q0 * sqrt(-1 + (self.q_wall / self.q0) ** 2))
        ) * sinh

    def __repr__(self):
        return (
            colored("Chris's q-factor", "light_blue")
            + f": q0={self.q0:.4g}, q_wall={self.q_wall:.4g}"
        )
