r"""
Events are used by SciPy's ``solve_ivp`` solver to locate where certain conditions
are met during the solving, and optionally terminate the solving after a certain
amount of times those conditions are found. More info 
`here <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html>`_.

Events are *function handles* of **analytical** funtions, and they trigger when their
``return`` expression is equal to 0. Since ``solve_ivp`` uses a root find algorithm to 
locate the exact position of the event, the expressions must be 0 **and** change sings 
around the root.

We can use multiple events, and the solver will return a list with the triggered positions
for each of them.

Example
-------

This event triggers whenever :math:`\theta` takes the value "theta0" and terminates 
after "terminal" triggers, which we can pass as arguments. The ``direction``
parameter only checks the **sign** of the direction.

.. code-block:: python

    def when_theta_trapped(theta0, terminal):

        def event(t, S):
            return S[0] - theta0

        event.terminal = terminal
        event.direction = 1

        return event

To use an event in a particle, you must pass it to its 
:py:meth:`~gcmotion.classes.particle.Particle.run` method as such:

.. code-block:: python

    >>> events = [gcm.events.when_theta(theta0, 8)]
    >>> cwp.run(events=events)

This event will stop the solver after :math:`\theta` was taken its initial
value 8 times, regardless if the particle turns out to be passing or trapped.
Note that this does **not** mean 8 periods, since the motion is multi-
periodic. 

.. rubric:: Availiable events
    :heading-level: 4

"""

from numpy import pi

from gcmotion.utils.logger_setup import logger


def when_theta(theta0, terminal, pole: int | float = pi / 2):
    r"""
    Triggers when :math:`\theta` mod :math:`2\pi` is equal to ``theta_0`` and
    terminates after ``terminal`` times (Starting position included).

    Can stop both trapped and passing orbits.

    Setting ``terminal=0`` makes the event non-terminal.

    .. note::
        By taking the mod(2π) of the particle's theta during the orbit calculation,
        we can find when it returns to its original value even if the particle is
        not trapped. That however creates problem in the (very common) case that our
        theta0 is 2κπ, since these points are a pole to the mod(2π) function and the
        event funtion is no longer analytical and continuous. To circumvent this problem,
        we can add any number to both runtime-theta and event theta, for example π/2, so
        the pole is now at κπ, and since we try to minimize their difference, this change
        does not affect the event locator. This number, reasonably called "pole" can
        be set to any number not-too-close to any initial theta0.

    Parameters
    ----------
    theta0 : float
        The :math:`\theta` value that triggers the event.
    terminal : int
        The event's termination number.
    pole : float, optional
        The modulo pole. Must be different than **ANY** of the
        initial theta0s. Defaults to π/2.

    Returns
    -------
    event : function handle
        The event function handle to be passed to solve_ivp.

    """

    # Warn about pole position
    if abs(pole - theta0) < 1e-3:
        string = "Warning. Pole dangerously close to an initial θ. Event might not trigger."
        print(string)
        logger.warning(string)

    def event(t, S):
        new = S[0] + pole
        new0 = theta0 + pole
        return new % (2 * pi) - new0  # * (new % (2 * pi) - new0)

    if terminal != 0:
        event.terminal = terminal

    return event


def when_psi(psi0, terminal):
    r"""
    Triggers when :math:`\psi` is equal to ``\psi0`` and
    terminates after ``terminal`` times (Starting position included).

    .. note::
        This event is used to stop the solver from calculating
        unecessarily long orbits, when 1 should be enough. Since :math:`\psi`
        seems to mostly be a well behaved periodic function and stays bounded
        even for the nastiest initial conditions, its a good enough critirion
        for stopping the solver. In case it fails, the solver will just keep
        going, so no harm is done.

    Setting ``terminal=0`` makes the event non-terminal.

    Parameters
    ----------
    psi0 : float
        The :math:`\theta` value that triggers the event.
    terminal : int
        The event's termination number.

    Returns
    -------
    event : function handle
        The event function handle to be passed to solve_ivp.
    """

    def event(t, S):
        return S[1] - psi0

    if terminal != 0:
        event.terminal = terminal

    return event
