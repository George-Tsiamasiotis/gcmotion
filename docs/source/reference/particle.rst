.. currentmodule:: gcmotion

=================
gcmotion.Particle
=================

.. autoclass:: Particle
   :members:

Solvers
=======

2 Solvers are available for calculating a particle's orbit, and each can be
useful in different circumstances.

Runge-Kutta 5(4)
----------------

This method is `SciPy's RK45 <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.RK45.html#scipy.integrate.RK45>`_ method. It is the default method when calling Particle.run().

The Particles :py:class:`InitialConditions` must have ``t_eval`` defined, since RK45 integrates over a specific timespan.

Events are also available for use with this solver, and more than one can be used at a time. The ``terminal`` parameter can be used to halt the integration after the event was trigger ``terminal`` times.

.. important::

   Even though multiple events with the same or different ``terminal`` can be used, they trigger completely intdependantly from one another. This means that we can't use events to halt the integration at a time where *more than one* events are triggered, and currently SciPy has no support for such method.

NPeriodSolver
-------------

The ``NPeriodSolver`` is a custom-made Solver basing `SciPy's OdeSolver <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.OdeSolver.html#scipy.integrate.OdeSolver>`_, and was created to circumvent the above problem. This is extremely useful, because it gives us the ability to stop the integration after the condition

.. math::

   \theta = \theta_0 \quad \text{and} \quad \psi = \psi_0

has been met N times, which amounts to exactly N completed orbit periods. This is derived from our Hamiltonian theory. 

.. note::

   Even though the actual conjugate variables are :math:`\psi` and :math:`P_\theta`, the condition for a full period still holds **if and only if** the :math:`\psi-P_\theta` relation is 1-1, which is always true since :math:`\partial P_\theta/\partial\psi> 0`

.. rubric:: Solver

.. autoclass:: gcmotion.scripts.orbits.nperiod_solver.NPeriodSolver
   :class-doc-from: class
   :show-inheritance:
   :member-order: bysource
   :members: _step_impl, period_check, last_step_recursion
   :private-members: _step_impl
