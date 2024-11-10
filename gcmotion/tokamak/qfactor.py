r"""
About q-Factor objects
----------------------

A q-Factor object is a class instance containing all the information about the 
q-factor profile of the system. It implements all the methods needed buy the
solver and other calculations, and is called automatically wherever required.

To add a new q-factor, simply copy-paste an already existing class
and fill the :py:meth:`~gcmotion.tokamak.qfactor.QFactor.solverqNU()` 
and :py:meth:`~gcmotion.tokamak.qfactor.QFactor.psipNU()` methods to fit
your q-factor. In case your q factor has extra parameters you want to pass 
as arguments, you must also create an 
:py:meth:`~gcmotion.tokamak.qfactor.QFactor.__init__()` method and declare 
them. A ``__repr__()`` method is also recommended for representing the system's
qfactor, but not enforced. To avoid errors, your class should inherit 
the :py:class:`~gcmotion.tokamak.qfactor.QFactor` class.

The general structure is this::

    class MyQFactor(QFactor):

        def __init__(self, *parameters):
            "Parameter setup."

        def solverqNU(self, psi):
            "Returns the value q(ψ)."
            return q

        def psipNU(self, psi):
            "Returns the value ψ_p(ψ)."
            return pisp

        def __repr__():
            "optional, but recommended"
            return string

.. note:: 
    The Qfactor's parameters should be Quantites. Conversions to [NU] and 
    intermediate values must be calculated in 
    :py:meth:`~gcmotion.tokamak.qfactor.QFactor.__init__()`.

.. admonition:: For developers

    For each attribute that is defined as a Quantity with SI units, another,
    "hidden" attribite is automatically defined as its magnitude. This hidden 
    attribute is then used for all the purely numerical calculations. 
    For example:

    .. code-block:: python

        def __init__(...):
            self.psi_wall = (B0 * a**2 / 2).to("Magnetic_flux")
            ...
            self._psi_wall = self.psi_wall.magnitude

    Here, :code:`self.psi_wall` is a Quantity with units of "Magnetic flux", however only
    :code:`self._psi_wall` is used inside the methods. Also, by defining it in 
    :code:`__init__()` we avoid having to retrieve its magnitude every time
    a method that needs it is called.

.. rubric:: The 'QFactor' Abstract Base Class

The base class that every other class inherits from. This class 
does nothing, it is only a template.

.. autoclass:: QFactor
    :member-order: bysource
    :members: __init__, solverqNU, psipNU

"""

import pint
import numpy as np
from math import sqrt, atan
from scipy.special import hyp2f1
from abc import ABC, abstractmethod

# Quantity alias for type annotations
type Quantity = pint.UnitRegistry.Quantity


class QFactor(ABC):
    r"""q Factor base class"""

    @abstractmethod
    def __init__(self):
        r"""Contains all the needed parameters."""

    @abstractmethod
    def solverqNU(self, psi: float) -> float:
        r"""Calculates :math:`q(\psi)`.
        Input and output must both be floats, in [NU].

        Used inside the solver.

        Parameters
        ----------
        psi : float
            Value of :math:`\psi` in [NU].

        Returns
        -------
        float
            Calculated :math:`q(\psi)`.

        """

    @abstractmethod
    def psipNU(self, psi: float | np.ndarray) -> float | np.ndarray:
        r"""Calculates :math:`\psi_p(\psi)`.
        Input and output must can be both floats or np.ndarrays, in [NU].

        Used in calculating :math:`\psi_p`'s time evolution from :math:`\psi`,
        :math:`\psi_{p,wall}`, etc.

        Parameters
        ----------
        psi : float | np.ndarray
            Value(s) of ψ.

        Returns
        -------
        float | np.ndarray
            Calculated :math:`\psi_p(\psi)`.

        """


# ====================================================


class Unity(QFactor):
    r"""Initializes an object q with :math:`q(\psi) = 1`
    and :math:`\psi_p=\psi`."""

    def __init__(self):
        pass

    def solverqNU(self, psi):
        return 1

    def psipNU(self, psi):
        return psi

    def __repr__(self):
        return "Unity"


class Parabolic(QFactor):
    r"""Initializes an object q with
    :math:`q(\psi) = q_0 + (q_{wall}-q_0)\bigg(\dfrac{\psi}{\psi_{wall}}\bigg)^2`

    :math:`\psi_p(\psi)` is calculated from:

    :math:`\psi_p(\psi) = \dfrac{\psi_{wall}}{\sqrt{q_0 (q_{wall}-q_0)}}\
    \arctan\bigg( \dfrac{\psi\sqrt{q_{wall}-q_0}}{\psi_{wall}\sqrt{q_0}} \bigg)`

    """

    def __init__(
        self,
        a: Quantity,
        B0: Quantity,
        q0: float,
        q_wall: float,
    ):
        r"""Parameters initialization.

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
        n : int
            Order of equillibrium (1: peaked, 2: round, 4: flat).
        """

        # SI Quantities
        self.B0 = B0.to("Tesla")
        self.a = a.to("meters")
        self.psi_wall = (B0 * a**2 / 2).to("Magnetic_flux")
        self.psi_wallNU = self.psi_wall.to("NUMagnetic_flux")

        # Unitless quantities, makes it a bit faster if defined here
        self._psi_wall = self.psi_wall.magnitude
        self._psi_wallNU = self.psi_wallNU.magnitude
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
        return "Parabolic: " + f"q0={self.q0:.4g}, q_wall={self.q_wall:.4g}."


class Hypergeometric(QFactor):
    r"""Initializes an object q with:
    :math:`q(\psi) = q_0\bigg\{ 1 + \bigg[ \bigg(\dfrac{q_{wall}}{q_0}\bigg)^n -1 \bigg] \
    \bigg( \dfrac{\psi}{\psi_{wall}} \bigg)^n \bigg\}^{1/n}`.
    
    :math:`\psi_p(\psi)` is calculated from:

    :math:`\psi_p(\psi) = \dfrac{\psi}{q_0} \phantom{1}_2 F_1\
    \bigg[ \dfrac{1}{n}, \dfrac{1}{n}, 1+\dfrac{1}{n},
    \bigg(1 - \bigg( \dfrac{q_{wall}}{q_0} \bigg)^n\bigg)
    \bigg( \dfrac{\psi}{\psi_{wall}} \bigg)^n \bigg]`,
    
    where :math:`\phantom{1}_2 F_1` the hypergeometric function.

    """

    def __init__(
        self,
        a: Quantity,
        B0: Quantity,
        q0: float,
        q_wall: float,
        n: int,
    ):
        r"""Parameters initialization.

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
        if isinstance(psi, (int, float)):
            return psi / self.q0 * float(hyp2f1(a, b, c, z))
        else:
            return psi / self.q0 * hyp2f1(a, b, c, z)

    def __repr__(self):
        return (
            "Hypergeometric: "
            + f"q0={self.q0:.4g}, q_wall={self.q_wall:.4g}, n={self.n:.4g}."
        )
