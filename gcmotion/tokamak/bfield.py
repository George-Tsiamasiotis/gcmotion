r"""
About Magnetic field objects
----------------------------

A Magnetic Field object is a class instance containing all the information 
about the magnetic field of the system. It implements all the methods needed 
buy the solver and other calculations, and is called automatically wherever required.

To add a new magnetic field, simply copy-paste an already existing class
and fill the :py:meth:`~gcmotion.tokamak.bfield.MagneticField.bigNU()` 
and :py:meth:`~gcmotion.tokamak.bfield.MagneticField.solverbNU()` methods to fit
your magnetic field. In case your field has extra parameters you want to pass 
as arguments, you must also create an 
:py:meth:`~gcmotion.tokamak.bfield.MagneticField.__init__()` method and declare 
them. A ``__repr__()`` method is also recommended for representing the system's
magnetic field, but not enforced. To avoid errors, your class should inherit 
the :py:class:`~gcmotion.tokamak.bfield.MagneticField` class.

The general structure is this::

    class MyMagneticField(MagneticField):

        def __init__(self, *parameters):
            "Parameter setup."
        
        def bigNU(self, psi, theta)
            return (b, g, i)
    
        def solverbNU(self, psi, theta): 

            b, g, i = self.bigNU(psi, theta)
            
            b_der = (b_der_psi, b_der_theta)
            currents = (i, g)
            currents_der = (i_der, g_der)
            
            return b, b_der, currents, currents_der

        def __repr__():
            "optional, but recommended"
            return string

.. note:: 
    The Magnetic Fields's parameters should be Quantites. Conversions to 
    [NU] and intermediate values must be calculated in 
    :py:meth:`~gcmotion.tokamak.bfield.MagneticField.__init__()`.

.. admonition:: For developers

    For each attribute that is defined as a Quantity with SI units, another,
    "hidden" attribite is automatically defined as its magnitude. This hidden 
    attribute is then used for all the purely numerical calculations. 
    For example:

    .. code-block:: python

        def __init__(...):
            self.i = i.to("Plasma_Current")
            self.iNU = i.to("NUPlasma_Current")
            ...
            self._i = self.i.magnitude
            self._iNU = self.iNU.magnitude

    Here, :code:`self.i` is a Quantity with units of "Plasma Current", however only
    :code:`self._i` is used inside the methods. Also, by defining it in 
    :code:`__init__()` we avoid having to retrieve its magnitude every time
    a method that needs it is called.
  
.. rubric:: The 'MagneticField' Abstract Base Class

The base class that every other class inherits from. 
This class does nothing, it is only a template.

.. autoclass:: MagneticField
    :member-order: bysource
    :members: __init__, bigNU, solverbNU

"""

import pint
import numpy as np
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
        r"""Calculates :math:`B(\psi, \theta), I(\psi, \theta), g(\psi, \theta)`.
        Input and output must be both floats or np.ndarrays, in [NU].

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
            The Calculated :math:`B(\psi, \theta), I(\psi, \theta), g(\psi, \theta)` value(s) in [NU].

        """

    @abstractmethod
    def solverbNU(
        self, psi: float, theta: float
    ) -> tuple[float, float, float]:
        r"""Calculates all the values needed by the solver:
        :math:`B,I,g` (by calling ``self.bigNU()``) and the derivatives
        :math:`\dfrac{\partial B}{\partial \psi}, \dfrac{\partial B}{\partial \theta}\
        \dfrac{\partial I}{\partial \psi}` and :math:`\dfrac{\partial g}{\partial \psi}`.

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
        3-tuple of floats
            Calculated values and derivatives in [NU].
        """


# =======================================================


class LAR(MagneticField):
    r"""Initializes the standard Large Aspect Ratio magnetic field."""

    def __init__(self, B0: Quantity, i: Quantity, g: Quantity):
        r"""Parameters initialization.

        Parameters
        -----------
        B0 : Quantity
            The strength of the magnetic field *on the magnetic axis*
            in [T].
        i : Quantity
            The toroidal current in units of [Plasma Current].
        g : Quantity
            The poloidal current in units of [Plasma Current].
        """

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
        b_der_psi = cos_theta / root
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
            "LAR: "
            + f"B0={self.B0:.4g~P}, I={self.i:.4g~P}, g={self.g:.4g~P}."
        )
