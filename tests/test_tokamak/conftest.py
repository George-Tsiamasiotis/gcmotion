r"""
Instantiates a Fixture Request with all availiable analytical qfactors, bfields
and efields, and 1 numerical in each case
"""

import pytest

import gcmotion as gcm


@pytest.fixture(scope="module")
def Q():
    r"""Creates a simple analytical Quantity Constructor."""

    Rnum = 1.65
    anum = 0.5
    B0num = 2
    species = "p"
    return gcm.QuantityConstructor(R=Rnum, a=anum, B0=B0num, species=species)


# =============================================================================


def unity_qfactor():
    r"""Creates the Unity qfactor."""
    return gcm.qfactor.Unity()


def parabolic_qfactor(Q):
    r"""Creates a Parabolic qfactor"""
    a = Q(0.5, "meters")
    B0 = Q(2, "Tesla")
    return gcm.qfactor.Parabolic(a, B0, q0=1.1, q_wall=3.8)


def hypergeometric_qfactor(Q):
    r"""Creates a Parabolic qfactor"""
    a = Q(0.5, "meters")
    B0 = Q(2, "Tesla")
    return gcm.qfactor.Hypergeometric(a, B0, q0=1.1, q_wall=3.8, n=2)


def precomputed_hypergeometric_qfactor(Q):
    r"""Creates a Parabolic qfactor"""
    a = Q(0.5, "meters")
    B0 = Q(2, "Tesla")
    return gcm.qfactor.PrecomputedHypergeometric(
        a, B0, q0=1.1, q_wall=3.8, n=2
    )


def numerical_qfactor():
    r"""Creates an example Numerical Qfactor."""
    return gcm.qfactor.SmartPositive()


"""
No need for other numerical qfactors, they work in the exact same way.
"""


@pytest.fixture(scope="module")
def qfactors(request, Q):
    r"""All availiable qfactors."""

    if request.param == "unity":
        return unity_qfactor()
    if request.param == "parabolic":
        return parabolic_qfactor(Q)
    if request.param == "hypergeometric":
        return hypergeometric_qfactor(Q)
    if request.param == "precomputed hypergeometric":
        return precomputed_hypergeometric_qfactor(Q)
    if request.param == "numerical":
        return numerical_qfactor()


# =============================================================================


def lar_bfield(Q):
    r"""Creates a simple LAR bfield"""
    B0 = Q(1, "Tesla")
    i = Q(0, "NUPlasma_current")
    g = Q(1, "NUPlasma_current")
    return gcm.bfield.LAR(B0=B0, i=i, g=g)


"""
We must check all numerical bfields, since we need to make sure we can locate
their Bmin and Bmax values needed for the Parabolas.
"""


def smart_pt_bfield():
    r"""Creates the Smart PT bfield"""
    return gcm.bfield.SmartPositive()


def smart_nt_bfield():
    r"""Creates the Smart NT bfield"""
    return gcm.bfield.SmartNegative()


def smart_nt_bfield2():
    r"""Creates the Smart NT2 bfield"""
    return gcm.bfield.SmartNegative2()


def dtt_pt_bfield():
    r"""Creates the DTT PT bfield"""
    return gcm.bfield.DTTPositive()


def dtt_nt_bfield():
    r"""Creates the DTT NT bfield"""
    return gcm.bfield.DTTNegative()


@pytest.fixture(scope="module")
def bfields(request, Q):
    r"""All availiable bfields."""

    if request.param == "lar":
        return lar_bfield(Q)
    if request.param == "smart_nt":
        return smart_pt_bfield()
    if request.param == "smart_pt":
        return smart_nt_bfield()
    if request.param == "smart_nt2":
        return smart_nt_bfield2()
    if request.param == "dtt_pt":
        return dtt_pt_bfield()
    if request.param == "dtt_nt":
        return dtt_nt_bfield()


# =============================================================================


def no_efield():
    r"""Creates a Nofield efield."""
    return gcm.efield.Nofield()


def radial_efield(Q):
    r"""Creates a Radial efield."""
    B0 = Q(2, "Tesla")
    a = Q(0.5, "meters")
    Ea = Q(73500, "Volts/meter")
    return gcm.efield.Radial(a, Ea, B0, peak=0.98, rw=1 / 50)


@pytest.fixture(scope="module")
def efields(request, Q):
    r"""All availiable analytical efields."""

    if request.param == "nofield":
        return no_efield()
    if request.param == "radial":
        return radial_efield(Q)
