r"""
Instantiates a Fixture Request with all availiable analytical qfactors and 1
numerical.
"""

import pytest

import gcmotion as gcm


@pytest.fixture(scope="module")
def Q():
    r"""Creates a simple analytical Quantity Constructor."""

    Rnum = 1.65
    anum = 0.5
    B0num = 1
    species = "p"
    return gcm.QuantityConstructor(R=Rnum, a=anum, B0=B0num, species=species)


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


@pytest.fixture(scope="module")
def qfactors(request, Q):
    r"""All availiable analytical qfactors."""

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
