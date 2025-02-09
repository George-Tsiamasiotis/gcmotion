.. currentmodule:: gcmotion

.. _initializers_docs:

############
Initializers
############

Since numerical equilibria have their own parameters for R and B0, stored in the respective dataset, we should use those to define our normalized units. Initializers do that job for as. They find those values in the dataset, create the Quantity Constructor and return it. They also create the correct Quantities for R and B0, which we can grab whenever we need them. However, the particle species is up to us.

Examples
--------

>>> import gcmotion as gcm
>>> 
>>> # Quantity Constructor
>>> species = "T"
>>> smart_init = gcm.SmartPositiveInit(species)
>>> Q = smart_init.QuantityConstructor()
>>>
>>> # Quantities, with units and everything
>>> R = smart_init.R
>>> a = smart_init.a
>>> B0 = smart_init.B0

Note
----

The "a" Quantity is purely decorative. It simply shows the horizontal distance of the magnetic axis to the outer wall edge (θ=0). In analytic equilibria we need it to define :math:`psi_wall`, but here :math:`psi_wall` is defined inside the dataset. In both cases, a isn't used in any further calculations.

Available Initializers
----------------------

A seperate initializer is needed for every numerical equilibrium, but they work in the exact same way

====================          ================================
Smart - Positive              :py:class:`SmartPositiveInit`
Smart - Negative              :py:class:`SmartNegativeInit`
Divertor - Negative           :py:class:`DivertorNegativeInit`
====================          ================================

.. autoclass:: SmartPositiveInit

.. autoclass:: SmartNegativeInit

.. autoclass:: DivertorNegativeInit
