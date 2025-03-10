import numpy as np
from math import isclose
from copy import deepcopy

import pytest
import gcmotion as gcm

# Quantity Constructor
Rnum = 1.65
anum = 0.5
B0num = 1
species = "D"
Q = gcm.QuantityConstructor(R=Rnum, a=anum, B0=B0num, species=species)

# Intermediate Quantities
R = Q(Rnum, "meters")
a = Q(anum, "meters")
B0 = Q(B0num, "Tesla")
i = Q(0, "NUPlasma_current")
g = Q(1, "NUPlasma_current")
Ea = Q(73500, "Volts/meter")

# Construct a Tokamak
tokamak = gcm.Tokamak(
    R=R,
    a=a,
    qfactor=gcm.qfactor.Hypergeometric(a, B0, q0=1.1, q_wall=3.8, n=2),
    bfield=gcm.bfield.LAR(B0=B0, i=i, g=g),
    efield=gcm.efield.Radial(a, Ea, B0, peak=0.98, rw=1 / 50),
)


# Test all 2^3 possibilities
@pytest.mark.parametrize("E", [Q(1e-4, "NUJoule"), None])
@pytest.mark.parametrize("mu", [Q(1e-4, "NUMagnetic_moment"), None])
@pytest.mark.parametrize("Pzeta", [Q(-0.07, "NUCanonical_momentum"), None])
def test_profile_instantiations(E, mu, Pzeta):
    profile = gcm.Profile(
        tokamak=tokamak,
        species=species,
        E=E,
        mu=mu,
        Pzeta=Pzeta,
    )
    profile.__repr__()
    profile.__str__()


def test_profile_properties(simple_profile, Q):
    r"""Tests that all properties are getting updated. Not gonna bother with
    checking values and units"""
    old_profile = deepcopy(simple_profile)
    new_profile = deepcopy(simple_profile)
    new_profile.E *= 2
    new_profile.mu *= 2
    new_profile.Pzeta *= 2

    assert old_profile.E != new_profile.E
    assert old_profile.ENU != new_profile.ENU
    assert old_profile.mu != new_profile.mu
    assert old_profile.muNU != new_profile.muNU
    assert old_profile.Pzeta != new_profile.Pzeta
    assert old_profile.PzetaNU != new_profile.PzetaNU

    old_profile = deepcopy(simple_profile)
    new_profile = deepcopy(simple_profile)
    new_profile.ENU *= 2
    new_profile.muNU *= 2
    new_profile.PzetaNU *= 2

    assert old_profile.E != new_profile.E
    assert old_profile.ENU != new_profile.ENU
    assert old_profile.mu != new_profile.mu
    assert old_profile.muNU != new_profile.muNU
    assert old_profile.Pzeta != new_profile.Pzeta
    assert old_profile.PzetaNU != new_profile.PzetaNU


def test_findPtheta(simple_profile, Q):
    """Tests profile.findPtheta for functionality and shape."""
    # Check Quantity shapes
    psi_float = Q(0.5, "NUmf")
    psi_array = Q(np.linspace(0.1, 0.5, 5), "mf")
    Ptheta_float = simple_profile.findPtheta(psi_float, "NUcanmom")
    Ptheta_array = simple_profile.findPtheta(psi_array, "canmom")
    assert isinstance(Ptheta_float.m, float)
    assert Ptheta_array.m.shape == psi_array.m.shape

    # Check float shapes
    psi_float = 0.5
    psi_array = np.linspace(0.1, 0.5, 5)
    Ptheta_float = simple_profile.findPtheta(psi_float, "No units")
    Ptheta_array = simple_profile.findPtheta(psi_array, "No units")
    assert isinstance(Ptheta_float, float)
    assert Ptheta_array.shape == psi_array.shape


def test_findEnergy(simple_profile, Q):
    """Tests profile.findEnergy for functionality and shape."""
    # Check Quantity shapes
    theta = 0
    psi_float = Q(0.5, "NUmf")
    psi_array = Q(np.linspace(0.1, 0.5, 5), "mf")
    E_float = simple_profile.findEnergy(psi_float, theta, "Joule")
    E_array = simple_profile.findEnergy(psi_array, theta, "keV")
    assert isinstance(E_float.m, float)
    assert E_array.m.shape == psi_array.m.shape

    # Check float shapes
    psi_float = 0.5
    psi_array = np.linspace(0.1, 0.5, 5)
    E_float = simple_profile.findEnergy(psi_float, theta, "Joule")
    E_array = simple_profile.findEnergy(psi_array, theta, "keV")
    assert isinstance(E_float, float)
    assert E_array.shape == psi_array.shape


@pytest.mark.parametrize("potential", [True, False])
def test_findPzeta(simple_profile, Q, potential):
    """Tests profile.findPzeta for functionality and shape."""
    theta = -np.pi  # avoid 0, can lead to ZeroDivision
    psi_float = Q(0.5, "NUmf")
    psi_array = Q(np.linspace(0.1, 0.5, 5), "mf")
    Pzeta_float = simple_profile.findPzeta(
        psi_float, theta, "NUcanmom", potential
    )
    Pzeta_array = simple_profile.findPzeta(
        psi_array, theta, "canmom", potential
    )
    assert isinstance(Pzeta_float.m, float)
    assert Pzeta_array.m.shape == psi_array.m.shape


@pytest.mark.parametrize("potential", [True, False])
def test_findmu(simple_profile, Q, potential):
    """Tests profile.findmu for functionality and shape."""
    theta = -np.pi  # avoid 0, can lead to ZeroDivision
    psi_float = Q(0.5, "NUmf")
    psi_array = Q(np.linspace(0.1, 0.5, 5), "mf")
    mu_float = simple_profile.findmu(
        psi_float, theta, "NUmagnetic_moment", potential
    )
    mu_array = simple_profile.findmu(
        psi_array, theta, "magnetic_moment", potential
    )
    assert isinstance(mu_float.m, float)
    assert mu_array.m.shape == psi_array.m.shape


def test_rhosign(simple_profile):
    # These must change is if simple_profile is changed
    psi_co = np.linspace(0.03, 0.05, 20)  # from plots
    psi_cu = np.linspace(0.002, 0.009, 20)  # from plots
    psi_un = np.concat((psi_co, psi_cu))

    assert simple_profile._rhosign(psi_co) == (False, True)
    assert simple_profile._rhosign(psi_cu) == (False, False)
    assert simple_profile._rhosign(psi_un)[0] is True
