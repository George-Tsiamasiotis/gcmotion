# pylint: skip-file
import pytest
import numpy as np
from math import isclose, pi
import gcmotion as gcm

tau = 2 * pi


def test_NPeriods_functionality(nperiods_particle):
    nperiods_particle.run(method="NPeriods", stop_after=3)


@pytest.mark.parametrize("stop_after", [*range(1, 4)])
def test_NPeriods_single_period(nperiods_particle, stop_after):
    r"""This one is kinda wacky. The last points are usually not very close to
    the initial, but the root finding algorithm always finds the correct
    periods. The error also increases with the number of periods
    """
    nperiods_particle.run(method="NPeriods", stop_after=stop_after)
    p = nperiods_particle

    rel_tol = 5e-2
    angles_abs_tol = tau / 10
    tolerances = {"rel_tol": rel_tol, "abs_tol": angles_abs_tol}

    # NOTE: Use events to avoid fmod pole errors
    theta_event = gcm.events.when_theta(root=p.theta0)
    zeta_event = gcm.events.when_theta(root=p.zeta0)

    assert isclose(theta_event(t=0, S=[p.theta[-1]]), 0, **tolerances)
    assert isclose(zeta_event(t=0, S=[p.zeta[-1]]), 0, **tolerances)

    assert isclose(p.psi0.m, p.psi.m[-1], rel_tol=rel_tol)
    assert isclose(p.rho0.m, p.rho.m[-1], rel_tol=rel_tol)
    assert isclose(p.psip0.m, p.psip.m[-1], rel_tol=rel_tol)
    assert isclose(p.Ptheta0.m, p.Ptheta.m[-1], rel_tol=rel_tol)
    assert isclose(p.Pzeta0.m, p.Pzeta.m[-1], rel_tol=rel_tol)
    assert isclose(p.psi0NU.m, p.psiNU.m[-1], rel_tol=rel_tol)
    assert isclose(p.rho0NU.m, p.rhoNU.m[-1], rel_tol=rel_tol)
    assert isclose(p.psip0NU.m, p.psipNU.m[-1], rel_tol=rel_tol)
    assert isclose(p.Ptheta0NU.m, p.PthetaNU.m[-1], rel_tol=rel_tol)
    assert isclose(p.Pzeta0NU.m, p.PzetaNU.m[-1], rel_tol=rel_tol)


@pytest.mark.parametrize("stop_after", [*range(1, 11)])
def test_NPeriods_stop_after(nperiods_particle, stop_after):
    nperiods_particle.run(method="NPeriods", stop_after=stop_after)
    assert len(nperiods_particle.t_periods) == stop_after


@pytest.mark.parametrize("stop_after", [*range(4, 11)])
def test_NPeriods_period_variance(nperiods_particle, stop_after):
    nperiods_particle.run(method="NPeriods", stop_after=stop_after)

    calculated_periods = np.diff(nperiods_particle.t_periods)
    variance = np.var(calculated_periods) / np.mean(calculated_periods)

    print(nperiods_particle)
    assert variance < 1e-3
