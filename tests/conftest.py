import gcmotion as gcm
import numpy as np
import pytest
import os
from pathlib import Path


@pytest.fixture(scope="session")
def datasets_exist():
    r"""Returns True if all datasets exists. pytest must run from rootdir."""
    dataset1 = Path("gcmotion/tokamak/reconstructed/smart_negative.nc")
    dataset2 = Path("gcmotion/tokamak/reconstructed/smart_positive.nc")
    dataset3 = Path("gcmotion/tokamak/reconstructed/divertor_negative.nc")
    return (
        os.path.isfile(dataset1)
        and os.path.isfile(dataset2)
        and os.path.isfile(dataset3)
    )


@pytest.fixture(scope="session")
def reference_path():
    r"""Path to API reference files relative to tests/reference/"""
    return Path("../../docs/source/reference/")


@pytest.fixture(scope="function")
def examples_path():
    r"""Path to examples folder relative to tests/reference/"""
    return Path("../../examples/")


@pytest.fixture(scope="session")
def Q():
    r"""An analytical Quantity Constructor"""
    return gcm.QuantityConstructor(R=1.65, a=0.5, B0=2, species="p")


@pytest.fixture(scope="session")
def simple_tokamak(Q):
    """Simplest tokamak object (B=LAR, q=Unity, E=Nofield)"""
    B0 = Q(2, "Tesla")
    i = Q(0, "NUPlasma_current")
    g = Q(1, "NUPlasma_current")

    return gcm.Tokamak(
        R=Q(1.65, "meters"),
        a=Q(0.5, "meters"),
        qfactor=gcm.qfactor.Unity(),
        bfield=gcm.bfield.LAR(B0=B0, i=i, g=g),
        efield=gcm.efield.Nofield(),
    )


@pytest.fixture(scope="session")
def simple_profile(simple_tokamak, Q):
    """Simplest profile object (B=LAR, q=Unity, E=Nofield)"""
    return gcm.Profile(
        tokamak=simple_tokamak,
        species="p",
        mu=Q(1e-5, "NUMagnetic_moment"),
        Pzeta=Q(-0.015, "NUCanonical_momentum"),
        E=Q(100, "keV"),
    )


@pytest.fixture(scope="session")
def simple_init(Q):
    r"""Simplest initial conditions object."""
    return gcm.InitialConditions(
        species="p",
        muB=Q(0.5, "keV"),
        theta0=1,
        zeta0=0,
        psi0=Q(0.5, "psi_wall"),
        Pzeta0=Q(0.02, "NUCanonical_momentum"),
        t_eval=Q(np.linspace(0, 1e-5, 10000), "seconds"),
    )


@pytest.fixture(scope="session")
def simple_particle(simple_tokamak, simple_init, Q):
    """Simplest particle object (B=LAR, q=Unity, E=Nofield)"""
    particle = gcm.Particle(tokamak=simple_tokamak, init=simple_init)
    particle.run()
    return particle
