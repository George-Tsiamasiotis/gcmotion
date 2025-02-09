###############
gcmotion.events
###############

Events are used by SciPy's ``solve_ivp`` solver to locate where certain
conditions are met during the solving, and optionally terminate the solving
after a certain amount of times those conditions are found. More info `here
<https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html>`_.

Events are *function handles* of **analytical** funtions, and they trigger when
their ``return`` expression is equal to 0. Since ``solve_ivp`` uses a root find
algorithm to locate the exact position of the event, the expressions must be 0
**and** change sings around the root.

We can use multiple events, and the solver will return a list with the
triggered positions for each of them.

Examples
--------

To use an event in a particle, you must pass it to its
:py:meth:`~gcmotion.classes.particle.Particle.run` method as such:

.. code-block:: python

   events = [gcm.events.when_theta(theta0, 8)]
   particle.run(events=events)

This event will stop the solver after :math:`\theta` was taken its initial
value 8 times, regardless if the particle turns out to be passing or trapped.
Note that this does **not** mean 8 periods, since the motion is multi-
periodic.

Available Events
================

==========           =====================================
when_theta           :py:func:`gcmotion.events.when_theta`
when_psi             :py:func:`gcmotion.events.when_psi`
when_zeta            :py:func:`gcmotion.events.when_zeta`
when_rho             :py:func:`gcmotion.events.when_rho`
==========           =====================================

.. currentmodule:: gcmotion

.. automodule:: gcmotion.events
   :members: when_theta, when_psi, when_zeta, when_rho
   :member-order: bysource
