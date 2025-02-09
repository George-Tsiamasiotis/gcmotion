.. _quantity_constructor:

.. currentmodule:: gcmotion

############################
gcmotion.QuantityConstructor
############################

A light reading of pint's `Defining Quantities <https://pint.readthedocs.io/en/stable/user/defining-quantities.html#using-the-constructor>`_ paragraph is recommended. Note that we use the notation `Q` instead of `Q_`.

Initializing the Quantity Constructor
=====================================

Since we like to use *Normalized Units* a lot, the :py:func:`QuantityConstructor` function extends pint's build-in units to include *NU*. Since the normalization is done with respect to the tokamak's major radius *R*, the magnetic field strength *B0* on the magnetic axis and on the particle's mass (which is defined through its species), these three values must be the first to be defined. To make our lives easier, we also define the minor radius *a* and pass it to to the QuantityConstructor, who will also create *psi_wall* as a unit of magnetic flux. This means that a quantity of `1psi_wall` is equal to the magnetic flux on the tokamak's last closed surface. That way we can define initial :math:`\psi_0` s with respect to the tokamak's wall, instead of guessing!

Here are some examples of how the Quantity Constructor `Q` is used in gcmotion:

>>> import gcmotion as gcm
>>> import numpy as np
>>> 
>>> # Define R, a, B0 and 'species' and lastly the Quantity Constructor 'Q'
>>> Rnum = 1.65 # meters
>>> anum = 0.5 # meters
>>> B0num = 1 # Tesla
>>> species = "D"
>>> Q = gcm.QuantityConstructor(R=Rnum, a=anum, B0=B0num, species=species)
>>>
>>> # Defining R, a and B0 correctly
>>> R = Q(Rnum, "meters")
>>> a = Q(anum, "meters")
>>> B0 = Q(B0num, "Tesla")

Now R, a and B0 are defined as *Quantities*, and can be used to create every sort of configuration. Mind you that their initial numeric values Rnum, anum and B0num **must** represent *meters* for the Constructor to define the units correctly, and you should reuse their values to define the Quantities, to avoid any errors. However, you could just as well do:

>>> R = Q(1650, "millimeters")

or

>>> R = Q(1.744051376e-16, "lightyears")

.. tip::

   All classes make sure that their parameters are instanciated in the correct units internally, no matter how you defined them, as long as they have the correct dimensionality.

Calling the constructor
-----------------------

Here is the Constructor's documentation.

.. autofunction:: QuantityConstructor

Defining complex Quantities
===========================

Sometimes we must define Quantities with units more complex than "meters" or "Tesla". For example, to define the strength of an electric field we can do:

>>> Ea = Q(100, "kiloVolts/meter")
>>> Ea
<Quantity(100, 'kilovolt / meter')>
>>> print(Ea)
100 kilovolt / meter

To define the toroidal and poloidal currents, we use the 

>>> i = Q(0, "Plasma_current")
>>> g = Q(0.5, "Plasma_current")

Here, "Plasma_current" is an alias to "Tesla * meters", which is the unit of the currents in SI. However, we would rather define g = 1 *in NU*. We will see how to do that in the next chapter.

To define a Quantity with units of *Magnetic_flux*, we would do the following:

>>> psi = Q(5, "Magnetic_flux")
>>> print(psi)
5 Magnetic_flux
>>> print(psi.to("Tesla * meters^2")) # doctest: +SKIP
5.0 meter ** 2 * tesla
>>> print(psi.to_base_units()) # doctest: +SKIP
5.0 kilogram * meter ** 2 / ampere / second ** 2

Here, "Magnetic_flux" is but an alias to "Tesla * meter^2"

We can convert "Magnetic_flux" to "psi_wall". That essentially shows us psi's position relative to the wall:

>>> print(psi_wall.to("psi_wall")) # doctest: +SKIP
40.0 psi_wall

With that logic, we can easily define initial conditions relative to the device's wall:

>>> psi0 = Q(0.85, "psi_wall")
>>> print(psi0.to("Tesla * meters ^ 2"))
0.10625 meter ** 2 * tesla

.. tip::

   Pint's string parser is very powerful: Not only it understands operations between units without needing a very strict syntax, it can also perform operations between Quantities and resolve the resulting dimentionality. Moreover, it offers a method of typechecking, since it doesn't allow for illegal operations (try adding R and B0!). However, avoid using abbreviations like 'mm' or 'kg', since 'kg' actually stands for 'kilogauss' instead of 'kilogram'. Finally, the constructor is case-insensitive and does not distinguish between pluralized forms (e.g. 'meter' and 'meters'.)






Converting to NU and back
=========================

You can also convert values to their corresponding NU units. For example:

>>> R = Q(Rnum, "meters")
>>> print(R)
1.65 meter
>>> print(round(R.to("NUmeters"))) # doctest: +SKIP
1.0 NUmeter
>>> print(round(B0.to("NUTesla"))) # doctest: +SKIP
1.0 NUtesla 

but

>>> print(a.to("NUmeters"))  # doctest: +SKIP
0.30303030303030304 NUmeter

This is a good sanity check to make sure the normalizations are defined correctly

To define the toroidal and poloidal currents in NU, we can similarly:

>>> i = Q(0, "NUPlasma_current")
>>> g = Q(1, "NUPlasma_current")
>>> print(g)
1 NUPlasma_current
>>> print(g.to("Tesla*meter")) # doctest: +SKIP
1.65 meter * tesla

Conversion Table
================

Here is a list of the availiable NU units:

==================== ==================================    ===========================================
Normalised Unit      Can be converted to:                  Description
==================== ==================================    ===========================================
NUsecond             second, hour, nanosecond, ...         Normalized time unit
NUw0                 Hz, Mhz, 1/s, ...                     Normalized frequency unit
NUmeter              meter, kilometer, micrometer, ...     Normalized length unit
NUJoule              Joule, eV, keV, ...                   Normalized energy unit
NUTesla              Tesla                                 Normalized magnetic flux density unit
NUvelocity           m/s, km/h, ...                        Normalized velocity unit
NUMagnetic_flux      Tesla * meter^2                       Normalized magnetic flux unit
NUCanonical_momentum Canonical_momentum, Joule * second    Normalized Canonical Momentum
NUPlasma_current     Tesla * meter                         Normalized plasma current unit
NUMagnetic_moment    Ampere * meter^2                      Normalized magnetic moment unit
NUVolts              Volts                                 Normalized electric potential strength unit
NUVolts_per_NUmeter  Volts/meter                           Normalized electric field strength unit
NUpsi_wall           psi_wall, Tesla * meter^2             Normalized psi_wall
==================== ==================================    ===========================================
