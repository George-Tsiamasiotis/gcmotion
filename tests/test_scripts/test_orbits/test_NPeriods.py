# pylint: skip-file
import pytest
import numpy as np
from math import isclose, pi
import gcmotion as gcm

from gcmotion.configuration.scripts_configuration import NPeriodSolverConfig

tau = 2 * pi


def test_NPeriods_functionality(nperiods_particle):
    nperiods_particle.run(method="NPeriods", stop_after=3)


# =============================================================================


@pytest.mark.parametrize("stop_after", [*range(1, 6)])
def test_NPeriods_stop_after_1to5_periods(nperiods_particle, stop_after):
    NPeriods_stop_after(nperiods_particle, stop_after)


@pytest.mark.parametrize("stop_after", [*range(1, 6)])
def test_NPeriods_final_points_1to5_periods(nperiods_particle, stop_after):
    NPeriods_final_points(nperiods_particle, stop_after)


@pytest.mark.parametrize("stop_after", [*range(1, 6)])
def test_NPeriods_period_variance_1to5_periods(nperiods_particle, stop_after):
    if stop_after == 1:
        return
    NPeriods_period_variance(nperiods_particle, stop_after)


@pytest.mark.slow
@pytest.mark.parametrize("stop_after", [*range(5, 11)])
def test_NPeriods_stop_after_5to10_periods(nperiods_particle, stop_after):
    NPeriods_stop_after(nperiods_particle, stop_after)


@pytest.mark.slow
@pytest.mark.parametrize("stop_after", [*range(5, 11)])
def test_NPeriods_final_points_5to10_periods(nperiods_particle, stop_after):
    NPeriods_final_points(nperiods_particle, stop_after)


@pytest.mark.slow
@pytest.mark.parametrize("stop_after", [*range(5, 11)])
def test_NPeriods_period_variance_5to10_periods(nperiods_particle, stop_after):
    if stop_after == 1:
        return
    NPeriods_period_variance(nperiods_particle, stop_after)


# =============================================================================


def NPeriods_final_points(nperiods_particle, stop_after):
    r"""Check that the last values of the orbit variables are actually close to
    the initial ones. 'How close' is defined in the configuration, which is
    alos used here.
    """
    nperiods_particle.run(method="NPeriods", stop_after=stop_after)
    p = nperiods_particle

    rel_tol = NPeriodSolverConfig.final_y_rtol
    angles_abs_tol = tau / 50
    tolerances = {"rel_tol": rel_tol, "abs_tol": angles_abs_tol}

    # NOTE: Use events to avoid fmod pole errors
    # NOTE: Do not check for z, since the motion is multiperiodic and it can
    # take any value
    theta_event = gcm.events.when_theta(root=p.theta0)

    assert isclose(theta_event(t=0, S=[p.theta[-1]]), 0, **tolerances)

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


def NPeriods_stop_after(nperiods_particle, stop_after):
    nperiods_particle.run(method="NPeriods", stop_after=stop_after)
    assert len(nperiods_particle.t_periods) == stop_after


def NPeriods_period_variance(nperiods_particle, stop_after):
    nperiods_particle.run(method="NPeriods", stop_after=stop_after)

    calculated_periods = np.diff(nperiods_particle.t_periods)
    variance = np.var(calculated_periods) / np.mean(calculated_periods)

    assert variance < 1e-3

    assert np.all(
        np.isclose(calculated_periods, np.mean(calculated_periods), rtol=1e-3)
    )
