.. _tokamak_configuration:

#############################
Tokamak Configuration Objects
#############################

The qfactor, bfield and efield modules contain Classes for creating instances of q-factors, magnetic fields and electric fields respectively. Those instaces contain all the information about the object, like their parameters, and implement a method for every needed calculation.

About Tokamak Configuration Objects
===================================

In general, a configuration object looks like this:

.. code-block:: python

    class MyQFactor(QFactor):

        def __init__(self, *parameters):
            """Parameter setup. Quantites and intermediate
            values are setup here"""

        def solverqNU(self, psi: float) -> float:
            """Calculates q(ψ). Input must be float, in [NU]"""
            return q

        def psipNU(self, psi, theta):
            """Calculates ψ_p(ψ). Input and output can be either floats or 
            np.ndarrays, in [NU]."""
            return psip

        def __repr__():
            """Object representation"""
            return str

The instantiation parameters must be Quantities in either SI or NU. The `__init__()` method makes sure they have the correct dimensionality and units, calculates intermediate values and stores their magnitudes in different fields. The rest of the methods deal only with purely numerical values and *not* with Quantities, since Quantities greatly impact performance.

The same methods are implemented for numerical objets, but they use splines to calculate the values.

All configruations inherit from :ref:`base classes<base-classes-documentation>` to ensure functionality.

.. toctree::
   :maxdepth: 2

   qfactor
   bfield
   efield

.. toctree::
   :hidden:

   initializers
   _base_classes 
