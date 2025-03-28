# pylint: skip-file
import pytest
import numpy as np
from math import isclose, pi
import gcmotion as gcm

tau = 2 * pi


def test_NPeriods_functionality(long_particle):
    long_particle.run(method="NPeriods", stop_after=3)


def test_NPeriods_single_period(long_particle):
    long_particle.run(method="NPeriods", stop_after=1)
    assert long_particle.orbit_percentage < 50

    rel_tol = 5e-2
    angle_abs_tol = tau / 10

    # fmt: off
    # NOTE: Use events to avoid fmod pole errors
    when_theta_event = gcm.events.when_theta(root=long_particle.theta0)
    when_zeta_event = gcm.events.when_zeta(root=long_particle.zeta0)

    assert isclose(when_theta_event(t=0, S=[long_particle.theta[-1]]), 0, rel_tol=rel_tol, abs_tol=angle_abs_tol)
    assert isclose(when_zeta_event(t=0, S=[0, 0, 0, long_particle.zeta[-1]]), 0, rel_tol=rel_tol, abs_tol=angle_abs_tol)

    assert isclose(long_particle.psi0.m, long_particle.psi.m[-1], rel_tol=rel_tol)
    assert isclose(long_particle.rho0.m, long_particle.rho.m[-1], rel_tol=rel_tol)
    assert isclose(long_particle.psip0.m, long_particle.psip.m[-1], rel_tol=rel_tol)
    assert isclose(long_particle.Ptheta0.m, long_particle.Ptheta.m[-1], rel_tol=rel_tol)
    assert isclose(long_particle.Pzeta0.m, long_particle.Pzeta.m[-1], rel_tol=rel_tol)
    assert isclose(long_particle.psi0NU.m, long_particle.psiNU.m[-1], rel_tol=rel_tol)
    assert isclose(long_particle.rho0NU.m, long_particle.rhoNU.m[-1], rel_tol=rel_tol)
    assert isclose(long_particle.psip0NU.m, long_particle.psipNU.m[-1], rel_tol=rel_tol)
    assert isclose(long_particle.Ptheta0NU.m, long_particle.PthetaNU.m[-1], rel_tol=rel_tol)
    assert isclose(long_particle.Pzeta0NU.m, long_particle.PzetaNU.m[-1], rel_tol=rel_tol) 
    # fmt: on


@pytest.mark.parametrize("stop_after", [*range(1, 11)])
def test_NPeriods_stop_after(
    simple_particle_single_period, long_particle, stop_after
):
    long_particle.run(method="NPeriods", stop_after=stop_after)
    assert len(long_particle.t_periods) == stop_after


@pytest.mark.parametrize("stop_after", [*range(4, 11)])
def test_NPeriods_period_variance(
    simple_particle_single_period, long_particle, stop_after
):
    long_particle.run(method="NPeriods", stop_after=stop_after)

    calculated_periods = np.diff(long_particle.t_periods)
    variance = np.var(calculated_periods) / np.mean(calculated_periods)

    print(long_particle)
    assert variance < 1e-3
