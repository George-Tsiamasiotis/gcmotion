import pytest
import numpy as np
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
    qfactor=gcm.qfactor.PrecomputedHypergeometric(
        a, B0, q0=1.1, q_wall=3.8, n=2
    ),
    bfield=gcm.bfield.LAR(B0=B0, i=i, g=g),
    efield=gcm.efield.Radial(a, Ea, B0, peak=0.98, rw=1 / 50),
)

# Create a Profile
profile = gcm.Profile(
    tokamak=tokamak,
    species=species,
    mu=Q(1e-4, "NUMagnetic_moment"),
    Pzeta=Q(-0.03, "NUCanonical_momentum"),
)

theta = 0
thetas = np.random.uniform((500, 500))

_psi = 0.05
psi = Q(_psi, "NUMagnetic_flux")

_psis = np.random.uniform((500, 500))
psis = Q(_psis, "NUmagnetic_flux")

# =============================== findPtheta ===============================


@pytest.mark.benchmark(group="findPtheta float")
def test_findPtheta_float(benchmark):
    benchmark(profile.findPtheta, psi, "NUCanonical_momentum")


@pytest.mark.benchmark(group="findPtheta float")
def test_findPtheta_float_to_units(benchmark):
    benchmark(profile.findPtheta, psi, "Joule * second")


@pytest.mark.benchmark(group="findPtheta float")
def test__findPthetaNU_float(benchmark):
    benchmark(profile._findPthetaNU, _psi)


@pytest.mark.benchmark(group="findPtheta array")
def test_findPtheta_array(benchmark):
    benchmark(profile.findPtheta, psis, "NUCanonical_momentum")


@pytest.mark.benchmark(group="findPtheta array")
def test_findPtheta_array_to_units(benchmark):
    benchmark(profile.findPtheta, psis, "Joule * second")


@pytest.mark.benchmark(group="findPtheta array")
def test__findPthetaNU_array(benchmark):
    benchmark(profile._findPthetaNU, _psis)


# =============================== findEnergy ===============================


@pytest.mark.benchmark(group="findEnergy float")
def test_findEnergy_float(benchmark):
    benchmark(profile.findEnergy, psi, theta, "NUJoule")


@pytest.mark.benchmark(group="findEnergy float")
def test_findEnergy_float_to_units(benchmark):
    benchmark(profile.findEnergy, psi, theta, "keV")


@pytest.mark.benchmark(group="findEnergy float")
def test__findEnergyNU_float(benchmark):
    benchmark(profile._findEnergyNU, _psi, theta, True)


@pytest.mark.benchmark(group="findEnergy array")
def test_findEnergy_array(benchmark):
    benchmark(profile.findEnergy, psis, thetas, "NUJoule")


@pytest.mark.benchmark(group="findEnergy array")
def test_findEnergy_array_to_units(benchmark):
    benchmark(profile.findEnergy, psis, thetas, "keV")


@pytest.mark.benchmark(group="findEnergy array")
def test__findEnergyNU_array(benchmark):
    benchmark(profile._findEnergyNU, _psis, thetas, True)
