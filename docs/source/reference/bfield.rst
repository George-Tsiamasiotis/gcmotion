.. _bfield_configuration:

###############
gcmotion.bfield
###############

.. currentmodule:: gcmotion.bfield

Here is a list of the availiable q-factor configurations:

============================     ============================
Large Aspect Ratio               :py:class:`LAR`
(Numerical) SmartPositve         :py:class:`SmartPositive`
(Numerical) SmartNegative        :py:class:`SmartNegative`
(Numerical) SmartNegative2       :py:class:`SmartNegative2`
(Numerical) DTTPositive          :py:class:`DTTPositive`
(Numerical) DTTNegative          :py:class:`DTTNegative`
============================     ============================

Their parameters are documented below.

.. rubric:: Examples

Creating an analytic Magnetic Field

>>> import gcmotion as gcm
>>>
>>> # Quantity Constructor
>>> Rnum = 1.6
>>> anum = 0.5
>>> B0num = 1
>>> species = "p"
>>> Q = gcm.QuantityConstructor(R=Rnum, a=anum, B0=B0num, species=species)
>>>
>>> # Intermediate values
>>> B0 = Q(B0num, "Tesla")
>>> i = Q(0, "NUPlasma_current")
>>> g = Q(1, "NUPlasma_current")
>>>
>>> # A magnetic field
>>> bfield = gcm.bfield.LAR(B0=B0, i=i, g=g)

Creating a numerical Magneticfield. Use the respective Initializer to grab the normalization constants from the dataset automatically. See :ref:`Initializers <initializers_docs>`

>>> import gcmotion as gcm
>>> 
>>> # Quantity Constructor
>>> species = "p"
>>> smart_init = gcm.SmartNegativeInit(species)
>>> Q = smart_init.QuantityConstructor()
>>> 
>>> # Intermediate Quantities
>>> R = smart_init.R
>>> a = smart_init.a
>>> B0 = smart_init.B0
>>>
>>> bfield=gcm.bfield.SmartNegative(),

*******
Methods
*******

The functions `bigNU` and `solverbNU` work identically in every class,
so I list their methods :ref:`here <base-classes-documentation>`. 


.. _available_bfields:

*************************
Available magnetic fields
*************************

.. rubric:: LAR

.. autoclass:: LAR
   :class-doc-from: class

.. rubric:: SmartPositive

.. autoclass:: SmartPositive
   :class-doc-from: class

.. rubric:: SmartNegative

.. autoclass:: SmartNegative
   :class-doc-from: class

.. rubric:: SmartNegative2

.. autoclass:: SmartNegative2
   :class-doc-from: class

.. rubric:: DTTPositive

.. autoclass:: DTTPositive
   :class-doc-from: class

.. rubric:: DTTNegative

.. autoclass:: DTTNegative
   :class-doc-from: class
