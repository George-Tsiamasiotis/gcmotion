"""
About Electric field objects
----------------------------

An Electric Field object is a class instance containing all the information 
about the electric field of the system. It implements all the methods needed 
buy the solver and other calculations, and is called automatically wherever required.

To add a new electic field, simply copy-paste an already existing class
and fill the :py:meth:`~gcmotion.tokamak.efield.ElectricField.solverPhiderNU()` ,
:py:meth:`~gcmotion.tokamak.efield.ElectricField.PhiNU()` and
:py:meth:`~gcmotion.tokamak.efield.ElectricField.Er()` methods to fit
your electric field. In case your field has extra parameters you want to pass 
as arguments, you must also create an 
:py:meth:`~gcmotion.tokamak.efield.ElectricField.__init__()` method and declare 
them. A ``__repr__()`` method is also recommended for representing the system's
electric field, but not enforced. To avoid errors, your class should inherit 
the :py:class:`~gcmotion.tokamak.efield.ElectricField` class.

The general structure is this::

    class MyElectricField(ElectricField):

        def __init__(self, *parameters):
            "Parameter setup."
    
        def solverPhiderNU(self, psi, theta): 
            return [Phi_der_psip, Phi_der_theta]
    
        def PhiNU(self, psi, theta):
            return Phi    

        def Er(self, psi, theta):
            return Er

        def __repr__():
            "optional, but recommended"
            return string

.. note:: 
    The Electric Fields's parameters should be Quantites. Conversions to 
    [NU] and intermediate values must be calculated in 
    :py:meth:`~gcmotion.tokamak.efield.ElectricField.__init__()`.

.. admonition:: For developers

    For each attribute that is defined as a Quantity with SI units, another,
    "hidden" attribite is automatically defined as its magnitude. This hidden 
    attribute is then used for all the purely numerical calculations. 
    For example:

    .. code-block:: python

        def __init__(...):
            self.rpeak = (self.a * self.peak).to("meters")
            ...
            self._rpeak = self.rpeak.magnitude

    Here, :code:`self.rpeak` is a Quantity with units of "meters", however only
    :code:`self._rpeak` is used inside the methods. Also, by defining it in 
    :code:`__init__()` we avoid having to retrieve its magnitude every time
    a method that needs it is called.

.. rubric:: The 'ElectricField' Abstract Base Class

The base class that every other class inherits from is ``ElectricField``. 
This class does nothing, it is only a template.

.. autoclass:: ElectricField
    :member-order: bysource
    :members: __init__, solverPhiderNU, PhiNU, Er

"""

import pint
import numpy as np
from scipy.special import erf
from math import sqrt, exp
from abc import ABC, abstractmethod

# Quantity alias for type annotations
type Quantity = pint.UnitRegistry.Quantity


class ElectricField(ABC):
    r"""Electric field base class."""

    def __init__(self):
        r"""Contains all the needed parameters"""

    @abstractmethod
    def solverPhiderNU(self, psi: float, theta: float) -> tuple[float, float]:
        r"""Calculates the derivatives of :math:`\Phi` with respect to
        :math:`\psi_p` and :math:`\theta`.
        Input and output must be floats, in [NU].

        Used inside the solver.

        .. warning::
            The derivatives are calculated with respect to :math:`\psi`,
            and **not** :math:`\psi_p`, which appear in the differential equations. 
            This is accounted for inside the solver by multiplying by :math:`q(\psi)`,
            since :math:`\dfrac{\partial f}{\partial \psi_p} = \dfrac{\partial f}{\partial \psi}\
            \dfrac{\partial \psi}{\partial \psi_p} = q\dfrac{\partial f}{\partial \psi}`

        Parameters
        ----------
        psi : float
            Value of :math:`\psi` in [NU].
        theta : float
            Value of :math:`\theta` in [NU].

        Returns
        -------
        2-tuple of floats
            Calculated derivatives in [NU].

        """

    @abstractmethod
    def PhiNU(
        self, psi: float | np.ndarray, theta: float | np.ndarray
    ) -> float | np.ndarray:
        r"""Calculates :math:`\Phi(\psi, theta)`.
        Input and output must can be both floats or np.ndarrays, in [NU].

        Used in energy contour plots.

        Parameters
        ----------
        psi : float | np.ndarray
            The :math:`\psi` value(s) in [NU].
        theta : float | np.ndarray
            The :math:`\psi` value(s).

        Returns
        -------
        float | np.ndarray
            The Calculated :math:`\Phi` value(s).

        """

    @abstractmethod
    def Er(self, psi: np.ndarray) -> np.ndarray:
        r"""Calculates the electric field strength in [NU].

        Used for plotting.

        Parameters
        ----------
        psi : np.ndarray
            The :math:`\psi` values.

        Returns
        -------
        np.ndarray
            1D numpy array with calculated :math:`E` values.

        """


# =======================================================


class Nofield(ElectricField):
    r"""Initializes an electric field of 0

    Exists to avoid compatibility issues.
    """

    def __init__(self):
        pass

    def solverPhiderNU(self, psi: float, theta: float) -> tuple[float, float]:
        return (0, 0)

    def PhiNU(
        self, psi: float | np.ndarray, theta: float | np.ndarray
    ) -> float | np.ndarray:

        if isinstance(psi, (int, float)):
            return 0
        elif isinstance(psi, np.ndarray):
            return 0 * psi

    def Er(self, psi: np.ndarray) -> np.ndarray:
        return 0 * psi

    def __repr__(self):
        return "No electric field,"


class Radial(ElectricField):
    r"""Initializes an electric field of the form:
    :math:`E(r) = -E_{peak}\exp\bigg[-\dfrac{(r-r_{peak})^2}{r_{w}^2})\bigg]`

    with

    :math:`\Phi(\psi) = E_{peak} \sqrt{\dfrac{\pi\psi_{wall}}{2}}\
    \bigg[ \text{erf} \bigg( \
    \dfrac{\sqrt{\psi} - \sqrt{\psi_{peak}}}{\sqrt{\psi_w}}\bigg) \
    + \text{erf} \bigg( \sqrt{\dfrac{\psi_{peak}}{\psi_w}} \bigg) \
    \bigg]`

    where

    :math:`\psi_w = r_w^2/2` and :math:`\psi_{peak} = r_{peak}^2/2`

    """

    def __init__(
        self,
        a: Quantity,
        Ea: Quantity,
        B0: Quantity,
        peak: float,
        rw: float,
    ):
        r"""Initializes the field's parameters.

        Parameters
        ----------

        a : Quantity
            The tokamak's minor radius in [m].
        Ea : Quantity
            The Electric field magnitude in [V/m].
        B0 : Quantity
            The Magnetic field's strength in [T].
        peak : float
            The Electric field's peak point with respect to
            :math:`\psi_{wall}`.
        rw : float
            The Electric field's waist width relative to :math:`r_{wall}`,
            defined as:
            :math:`r_{waist} = \alpha\cdot \text{rw}`.

        """

        # SI Quantities
        self.a = a.to("meters")
        self.Ea = Ea.to("Volts/meter")
        self.B0 = B0.to("Tesla")
        self.rpeak = (self.a * peak).to("meters")
        self.rw = (self.a * rw).to("meters")  # waist, not wall
        self.psi_peak = (self.B0 * self.rpeak**2 / 2).to("Magnetic_flux")
        self.psi_waist = (self.B0 * self.rw**2 / 2).to("Magnetic_flux")

        # [NU] Conversions for the derivatives:
        self.EaNU = self.Ea.to("NUVolts_per_NUmeter")
        self.psi_peakNU = self.psi_peak.to("NUMagnetic_flux")
        self.psi_waistNU = self.psi_waist.to("NUMagnetic_flux")
        self.rpeakNU = self.rpeak.to("NUmeters")
        self.rwNU = self.rw.to("NUmeters")

        # Unitless quantities, makes it a bit faster if defined here
        for key, value in self.__dict__.copy().items():
            self.__setattr__("_" + key, value.magnitude)

        self._sr_psi_peak = sqrt(self._psi_peak)
        self._sr_psi_waist = sqrt(self._psi_waist)
        self._sr_psi_peakNU = sqrt(self._psi_peakNU)
        self._sr_psi_waistNU = sqrt(self._psi_waistNU)

    def solverPhiderNU(self, psi: float, theta: float) -> tuple[float, float]:
        Phi_der_psi = (
            self._EaNU
            / (sqrt(2 * psi))
            * exp(
                -((sqrt(psi) - self._sr_psi_peakNU) ** 2) / self._psi_waistNU
            )
        )
        Phi_der_theta = 0

        return (Phi_der_psi, Phi_der_theta)

    def PhiNU(
        self, psi: float | np.ndarray, theta: float | np.ndarray
    ) -> float | np.ndarray:
        Phi = (
            self._EaNU
            * np.sqrt(np.pi * self._psi_waistNU / 2)
            * (
                erf(
                    (np.sqrt(psi) - self._sr_psi_peakNU) / self._sr_psi_waistNU
                )
                + erf(self._sr_psi_peakNU / self._sr_psi_waistNU)
            )
        )
        return Phi

    def Er(self, psi: np.ndarray) -> np.ndarray:
        r = np.sqrt(2 * psi)
        E = -self._EaNU * np.exp(-((r - self._rpeakNU) ** 2) / self._rwNU**2)
        return E

    def __repr__(self):
        return (
            "Radial: "
            + f"Ea={self.Ea:.4g~P}, peak={self.rpeak:.4g~P}, rw={self.rw:.4g~P}."
        )
