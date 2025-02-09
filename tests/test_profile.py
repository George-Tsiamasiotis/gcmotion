import numpy as np
from math import isclose


def test_findPtheta(simple_profile, Q):
    """Tests profile.findPtheta for functionality and shape."""
    psi_float = Q(0.5, "NUmf")
    psi_array = Q(np.linspace(0.1, 0.5, 5), "mf")
    Ptheta_float = simple_profile.findPtheta(psi_float, "NUcanmom")
    Ptheta_array = simple_profile.findPtheta(psi_array, "canmom")
    assert isinstance(Ptheta_float.m, float)
    assert Ptheta_array.m.shape == psi_array.m.shape


def test_findEnergy(simple_profile, Q):
    """Tests profile.findEnergy for functionality and shape."""
    theta = 0
    psi_float = Q(0.5, "NUmf")
    psi_array = Q(np.linspace(0.1, 0.5, 5), "mf")
    E_float = simple_profile.findEnergy(psi_float, theta, "Joule")
    E_array = simple_profile.findEnergy(psi_array, theta, "keV")
    assert isinstance(E_float.m, float)
    assert E_array.m.shape == psi_array.m.shape


def test_findPzeta(simple_profile, Q):
    """Tests profile.findPzeta for functionality and shape."""
    theta = -np.pi  # avoid 0, can lead to ZeroDivision
    psi_float = Q(0.5, "NUmf")
    psi_array = Q(np.linspace(0.1, 0.5, 5), "mf")
    Pzeta_float = simple_profile.findPzeta(psi_float, theta, "NUcanmom")
    Pzeta_array = simple_profile.findPzeta(psi_array, theta, "canmom")
    assert isinstance(Pzeta_float.m, float)
    assert Pzeta_array.m.shape == psi_array.m.shape


def test_findmu(simple_profile, Q):
    """Tests profile.findmu for functionality and shape."""
    theta = -np.pi  # avoid 0, can lead to ZeroDivision
    psi_float = Q(0.5, "NUmf")
    psi_array = Q(np.linspace(0.1, 0.5, 5), "mf")
    mu_float = simple_profile.findmu(psi_float, theta, "NUmagnetic_moment")
    mu_array = simple_profile.findmu(psi_array, theta, "magnetic_moment")
    assert isinstance(mu_float.m, float)
    assert mu_array.m.shape == psi_array.m.shape


def test_updatePzeta(simple_profile, Q):
    Pzeta_float = 0.2
    Pzeta_quant = Q(0.3, "NUcanonical_momentum")
    simple_profile._update_Pzeta(Pzeta_float)
    assert isclose(Pzeta_float, simple_profile.PzetaNU.m)
    simple_profile._update_Pzeta(Pzeta_quant)
    assert isclose(Pzeta_quant.m, simple_profile.PzetaNU.m)


def test_profile_repr_str(simple_profile):
    simple_profile.__repr__()
    simple_profile.__str__()
