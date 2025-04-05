import pytest
import numpy as np
import gcmotion as gcm

from gcmotion.scripts.orbits.nperiod_solver import ESCAPED_WALL, TIMED_OUT


def test_NPeriods_timeout_halt(simple_tokamak, Q):
    r"""Tests that the particle takes too long to complete a period. The
    initial conditions are set very close to the separatrix."""

    init = gcm.InitialConditions(
        species="p",
        muB=Q(50, "keV"),
        theta0=np.pi,
        zeta0=0,
        psi0=Q(0.01986043, "NUmf"),
        Pzeta0=Q(-0.02, "NUcanmom"),
    )
    particle = gcm.Particle(tokamak=simple_tokamak, init=init)

    with pytest.warns(expected_warning=UserWarning, match=TIMED_OUT):
        particle.run(method="NPeriods", stop_after=1)

    assert not hasattr(particle, "t_solve")
    assert particle.flags == {
        "succeded": False,
        "timed_out": True,
        "escaped_wall": False,
    }


def test_NPeriods_escaped_wall_halt(simple_tokamak, Q):
    r"""Tests that the particle escapes the wall."""

    init = gcm.InitialConditions(
        species="p",
        muB=Q(1e-4, "NUMagnetic_moment"),
        theta0=np.pi,
        zeta0=0,
        psi0=Q(0.9, "psi_wall"),
        Pzeta0=Q(-0.02, "NUcanmom"),
    )
    particle = gcm.Particle(tokamak=simple_tokamak, init=init)

    with pytest.warns(expected_warning=UserWarning, match=ESCAPED_WALL):
        particle.run(method="NPeriods", stop_after=1)

    assert not hasattr(particle, "t_solve")
    assert particle.flags == {
        "succeded": False,
        "timed_out": False,
        "escaped_wall": True,
    }
