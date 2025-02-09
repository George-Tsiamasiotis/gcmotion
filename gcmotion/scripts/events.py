r"""
======
Events
======

Defines the event constructor, which creates SciPy-ready event functions. Note
that since the solver only solves for "theta", "psi", "zeta" and "rho", these
are the only ones that can be tracked. The Canonical momenta and psip are
calculated afterwards.
"""

from math import sin
from pint import Quantity

_variable_indeces = {
    "theta": 0,
    "psi": 1,
    "zeta": 2,
    "rho": 3,
}


class _Event:
    r"""Constructs an event function handle, with the signature Scipy expects.

    Might be more pythonic to do this with a decorator but this seems more
    readable.

    Parameters
    ----------
    variable: {"theta", "psi", "zeta", "rho"}
        The tracked variable.
    terminal: int
        After how many occurences to terminate the solver. As SciPy defines it:
        terminal=0: Do not terminate the solver (still tracks occurences).
        terminal>=1: Stop the solver after "terminal" occurences
    direction: float
        Direction of a zero crossing. If direction is positive, event will only
        trigger when going from negative to positive, and vice versa if
        direction is negative. If 0, then either direction will trigger event.
        Implicitly 0 if not assigned. Note that this is not necessarily the
        direction of zero crossing of the variable itself, but the event's
        analytical expression. Defaults to 0.

    """

    def __init__(self, variable: str, terminal: int, direction: float):
        self.index = _variable_indeces[variable]
        self.terminal = terminal
        self.direction = direction
        self.name = "when_" + variable
        self.variable = variable

    def __call__(self, root: float | Quantity):
        r"""Sets up the event and its attributes and returns it."""

        if isinstance(root, Quantity):
            if self.variable == "psi":
                root = root.to("NUMagnetic_flux").m
            elif self.variable == "rho":
                root = root.to("NUmeters").m

        # PERF: This function seems to always work for both bounded and
        # unbounded variables. It treats all variables as angles, which means
        # that a lost particle with a rising psi will also trigger. This
        # doesn't make much sense but we don't really care about such particles
        # do we... Apart from theta and zeta, we expect the variables to be
        # bounded.
        def event(t, S):
            # OPTIM:
            return sin(0.5 * (S[self.index] - root))

        event.root = root
        event.name = self.name
        event.terminal = self.terminal
        event.direction = self.direction
        event.variable = self.variable

        return event


def when_theta(
    root,
    terminal: int = 0,
    direction: float = 0,
):
    r"""Occurs at :math:`\theta` = ``root``.

    Parameters
    ----------
    root: int | Quantity
        The :math:`\theta` value to be tracked.
    terminal: int, optional
        After how many occurences to terminate the solver. As SciPy defines it:

        #. terminal=0: Do not terminate the solver (still tracks occurences).

        #. terminal>=1: Stop the solver after "terminal" occurences

        Defaults to 0.
    direction: float, optional
        Direction of a zero crossing. If direction is positive, event will only
        trigger when going from negative to positive, and vice versa if
        direction is negative. If 0, then either direction will trigger event.
        Implicitly 0 if not assigned. Note that this is not necessarily the
        direction of zero crossing of the variable itself, but the event's
        analytical expression. Defaults to 0.

    """
    event = _Event(
        variable="theta",
        terminal=terminal,
        direction=direction,
    )(root)
    return event


def when_psi(
    root,
    terminal: int = 0,
    direction: float = 0,
):
    r"""Occurs at :math:`\psi` = ``root``.

    Parameters
    ----------
    root: int | Quantity
        The :math:`\psi` value to be tracked.
    terminal: int, optional
        After how many occurences to terminate the solver. As SciPy defines it:

        #. terminal=0: Do not terminate the solver (still tracks occurences).

        #. terminal>=1: Stop the solver after "terminal" occurences

        Defaults to 0.
    direction: float, optional
        Direction of a zero crossing. If direction is positive, event will only
        trigger when going from negative to positive, and vice versa if
        direction is negative. If 0, then either direction will trigger event.
        Implicitly 0 if not assigned. Note that this is not necessarily the
        direction of zero crossing of the variable itself, but the event's
        analytical expression. Defaults to 0.

    """
    event = _Event(
        variable="psi",
        terminal=terminal,
        direction=direction,
    )(root)
    return event


def when_zeta(
    root,
    terminal: int = 0,
    direction: float = 0,
):
    r"""Occurs at :math:`\zeta` = ``root``.

    Parameters
    ----------
    root: int | Quantity
        The :math:`\zeta` value to be tracked.
    terminal: int, optional
        After how many occurences to terminate the solver. As SciPy defines it:

        #. terminal=0: Do not terminate the solver (still tracks occurences).

        #. terminal>=1: Stop the solver after "terminal" occurences

        Defaults to 0.
    direction: float, optional
        Direction of a zero crossing. If direction is positive, event will only
        trigger when going from negative to positive, and vice versa if
        direction is negative. If 0, then either direction will trigger event.
        Implicitly 0 if not assigned. Note that this is not necessarily the
        direction of zero crossing of the variable itself, but the event's
        analytical expression. Defaults to 0.

    """
    event = _Event(
        variable="zeta",
        terminal=terminal,
        direction=direction,
    )(root)
    return event


def when_rho(
    root,
    terminal: int = 0,
    direction: float = 0,
):
    r"""Occurs at :math:`\rho_{||}` = ``root``.

    Parameters
    ----------
    root: int | Quantity
        The :math:`\rho_{||}` value to be tracked.
    terminal: int, optional
        After how many occurences to terminate the solver. As SciPy defines it:

        #. terminal=0: Do not terminate the solver (still tracks occurences).

        #. terminal>=1: Stop the solver after "terminal" occurences

        Defaults to 0.
    direction: float, optional
        Direction of a zero crossing. If direction is positive, event will only
        trigger when going from negative to positive, and vice versa if
        direction is negative. If 0, then either direction will trigger event.
        Implicitly 0 if not assigned. Note that this is not necessarily the
        direction of zero crossing of the variable itself, but the event's
        analytical expression. Defaults to 0.

    """
    event = _Event(
        variable="rho",
        terminal=terminal,
        direction=direction,
    )(root)
    return event
