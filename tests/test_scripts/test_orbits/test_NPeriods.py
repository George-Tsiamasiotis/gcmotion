import pytest
import numpy as np
from math import isclose, pi
import gcmotion as gcm

from gcmotion.configuration.scripts_configuration import NPeriodSolverConfig

tau = 2 * pi


@pytest.mark.parametrize(
    "nperiods_particle",
    ["trapped", "passing"],
    indirect=True,
)
def test_NPeriods_functionality(nperiods_particle):
    nperiods_particle.run(method="NPeriods", stop_after=3)
    assert nperiods_particle.flags == {
        "succeded": True,
        "timed_out": False,
        "escaped_wall": False,
    }


@pytest.mark.parametrize(
    "nperiods_particle",
    ["trapped", "passing"],
    indirect=True,
)
@pytest.mark.parametrize("stop_after", [*range(1, 6)])
def test_NPeriods_final_points(nperiods_particle, stop_after):
    r"""Check that the last values of the orbit variables are actually close to
    the initial ones. 'How close' is defined in the configuration, which is
    also used here.
    """
    nperiods_particle.run(method="NPeriods", stop_after=stop_after)
    p = nperiods_particle

    atol = 1e-4  # Needed for small values
    rtol = NPeriodSolverConfig.final_y_rtol
    angles_abs_tol = tau / 50

    # NOTE: Use events to avoid fmod pole errors
    # NOTE: Do not check for z, since the motion is multiperiodic and it can
    # take any value
    theta_event = gcm.events.when_theta(root=p.theta0)

    tolerances = {"rel_tol": rtol, "abs_tol": angles_abs_tol}
    assert isclose(theta_event(t=0, S=[p.theta[-1]]), 0, **tolerances)

    assert isclose(p.psi0.m, p.psi.m[-1], rel_tol=rtol, abs_tol=atol)
    assert isclose(p.rho0.m, p.rho.m[-1], rel_tol=rtol, abs_tol=atol)
    assert isclose(p.psip0.m, p.psip.m[-1], rel_tol=rtol, abs_tol=atol)
    assert isclose(p.Ptheta0.m, p.Ptheta.m[-1], rel_tol=rtol, abs_tol=atol)
    assert isclose(p.Pzeta0.m, p.Pzeta.m[-1], rel_tol=rtol, abs_tol=atol)
    assert isclose(p.psi0NU.m, p.psiNU.m[-1], rel_tol=rtol, abs_tol=atol)
    assert isclose(p.rho0NU.m, p.rhoNU.m[-1], rel_tol=rtol, abs_tol=atol)
    assert isclose(p.psip0NU.m, p.psipNU.m[-1], rel_tol=rtol, abs_tol=atol)
    assert isclose(p.Ptheta0NU.m, p.PthetaNU.m[-1], rel_tol=rtol, abs_tol=atol)
    assert isclose(p.Pzeta0NU.m, p.PzetaNU.m[-1], rel_tol=rtol, abs_tol=atol)

    assert len(p.t_solve) >= NPeriodSolverConfig.min_step_num

    assert p.flags == {
        "succeded": True,
        "timed_out": False,
        "escaped_wall": False,
    }


@pytest.mark.parametrize(
    "nperiods_particle",
    ["trapped", "passing"],
    indirect=True,
)
@pytest.mark.parametrize("stop_after", [*range(1, 6)])
def test_NPeriods_stop_after(nperiods_particle, stop_after):
    r"""Test that the solver does indeed stop after `stop_after` periods."""
    nperiods_particle.run(method="NPeriods", stop_after=stop_after)
    assert len(nperiods_particle.t_periods) == stop_after
    assert nperiods_particle.flags == {
        "succeded": True,
        "timed_out": False,
        "escaped_wall": False,
    }


@pytest.mark.parametrize(
    "nperiods_particle",
    ["trapped", "passing"],
    indirect=True,
)
@pytest.mark.parametrize("stop_after", [*range(2, 6)])
def test_NPeriods_period_variance(nperiods_particle, stop_after):
    r"""Test that the calculated periods are close enough."""
    nperiods_particle.run(method="NPeriods", stop_after=stop_after)

    periods = np.diff(nperiods_particle.t_periods)
    variance = np.var(periods) / np.mean(periods)

    assert variance < 1e-3

    assert np.all(np.isclose(periods, np.mean(periods), rtol=1e-3))
    assert nperiods_particle.flags == {
        "succeded": True,
        "timed_out": False,
        "escaped_wall": False,
    }
