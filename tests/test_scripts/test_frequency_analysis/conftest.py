import pytest
import gcmotion as gcm


def freq_tokamak_analytical():
    """Simplest tokamak object (B=LAR, q=Unity, E=Nofield)"""
    Q = gcm.QuantityConstructor(R=1.65, a=0.5, B0=1, species="p")
    B0 = Q(1, "Tesla")
    i = Q(0, "NUPlasma_current")
    g = Q(1, "NUPlasma_current")

    return gcm.Tokamak(
        R=Q(1.65, "meters"),
        a=Q(0.5, "meters"),
        qfactor=gcm.qfactor.Unity(),
        bfield=gcm.bfield.LAR(B0=B0, i=i, g=g),
        efield=gcm.efield.Nofield(),
    )


def freq_tokamak_numerical():
    """Simplest tokamak object with numerical data"""
    species = "p"
    init = gcm.DTTNegativeInit(species)
    # Q = init.QuantityConstructor()
    R = init.R
    a = init.a

    return gcm.Tokamak(
        R=R,
        a=a,
        qfactor=gcm.qfactor.DTTNegative(),
        bfield=gcm.bfield.DTTNegative(),
        efield=gcm.efield.Nofield(),
    )


@pytest.fixture(scope="session")
def freq_tokamaks(request):
    r"""All tokamaks used in frequency analysis"""

    if request.param == "analytical":
        return freq_tokamak_analytical()
    if request.param == "numerical":
        return freq_tokamak_numerical()
