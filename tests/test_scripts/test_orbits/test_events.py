import pytest

from pint.registry import Quantity
from math import isclose, tau, fmod


@pytest.mark.parametrize(
    "events",
    ["when_theta", "when_psi", "when_zeta", "when_rho"],
    indirect=True,
)
@pytest.mark.parametrize(
    "terminal",
    [1, 2, 3, 4],
    indirect=False,
)
def test_events(events_particle, events, terminal):
    r"""Test that all events halt the integration"""

    events_particle.run(method="RK45", events=[events])
    last_value = getattr(events_particle, events.variable)[-1].m

    variable0 = getattr(events_particle, events.variable + "0")
    if isinstance(variable0, Quantity):
        variable0 = variable0.m

    # NOTE: We have to check both the root and its explementary angle (that is,
    # θ and 2π-θ), due to the pole of fmod at 2π.
    assert isclose(
        fmod(variable0, tau),
        fmod(last_value, tau),
        abs_tol=0.08,
    ) or isclose(
        fmod(variable0, tau),
        tau - abs(fmod(last_value, tau)),
        abs_tol=0.08,
    )
    assert events_particle.method == "RK45"
    assert events_particle.orbit_percentage < 99  # Avoid floating point errors
    assert events_particle.t_events.m.flatten().shape[0] == events.terminal
