.. _main_namespace:

###########################
The main GCMotion namespace
###########################

.. currentmodule:: gcmotion

The main GCMotion namespace offers the creation of *entities*, such as a tokamak *Profile*, a *Particle* or a *Collection* of *Particles*, as well as tokamak configuration objects, such as a *QFactor*, *MagneticField* and *ElectricField*:

Constructing *Quantities*
=========================

All variables that represent physical quantities (except from angles) are defind as **Quantities**, using the `pint <https://pint.readthedocs.io/en/stable/>`_ libary. See how you can define them here:

.. toctree::
   :maxdepth: 2

   quantity

Tokamak Configuration
=====================

A :py:class:`Tokamak` entity, apart from its major and minor radii, requires a q-factor, a magnetic and an electric field to be defined:

============================    =========================
:ref:`qfactor_configuration`    Creates a q-factor
:ref:`bfield_configuration`     Creates a Magnetic field
:ref:`efield_configuration`     Creates an Electric field
============================    =========================

See also:

.. toctree::
   :maxdepth: 1

   tokamak
   initializers

Essential Entities
==================

.. autosummary::
   :toctree: generated/

   Tokamak
   Profile
   InitialConditions
   Particle

Utilities
=========

.. autosummary::
   :toctree: generated/

   get_size

Scripts
=======

.. toctree:: 
   :maxdepth: 1

   events
