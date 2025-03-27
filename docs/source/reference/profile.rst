.. currentmodule:: gcmotion

================
gcmotion.Profile
================

.. autoclass:: Profile
   :class-doc-from: class
   :members: findEnergy, findPtheta, 


.. note::

   The private methods ``_findPthetaNU`` and ``_findEnergyNU`` are also available. They inputs and outputs are pure numbers in NU, and therefore are considerably faster. The public methods :py:meth:`~Profile.findEnergy` and :py:meth:`~Profile.findPtheta` use these too internally as well, and only handle the units.

.. autosummary::

   Profile._findPthetaNU
   Profile._findEnergyNU

