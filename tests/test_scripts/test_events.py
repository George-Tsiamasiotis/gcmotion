import pytest

from pint import Quantity
from math import isclose, tau, fmod


@pytest.mark.parametrize(
    "events",
    ["when_theta", "when_psi", "when_zeta", "when_rho"],
    indirect=True,
)
@pytest.mark.parametrize(
    "terminal",
    [2, 3, 4],
    indirect=False,
)
def test_events(simple_particle, events, terminal):
    simple_particle.run(events=[events])
    # gcm.plot.particle_evolution(simple_particle, which="theta, psi, zeta, rho")
    last_value = getattr(simple_particle, events.variable)[-1].m
    variable0 = getattr(simple_particle, events.variable + "0")
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
    assert simple_particle.orbit_percentage < 100
    assert simple_particle.t_events.m.flatten().shape[0] == events.terminal
