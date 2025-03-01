import pytest
import numpy as np
import gcmotion as gcm


@pytest.fixture(scope="module")
def hyper(Q):
    r"""Creates a Hypergeometric qfactor"""
    a = Q(0.5, "meters")
    B0 = Q(2, "Tesla")
    return gcm.qfactor.Hypergeometric(a, B0, q0=1.1, q_wall=3.8, n=2)


@pytest.fixture(scope="module")
def hyper_pre(Q):
    r"""Creates a Precomputed Hypergeometric qfactor"""
    a = Q(0.5, "meters")
    B0 = Q(2, "Tesla")
    return gcm.qfactor.PrecomputedHypergeometric(
        a, B0, q0=1.1, q_wall=3.8, n=2
    )


def test_analytic_and_precomputed_hyp3f1(Q, hyper, hyper_pre):
    psis = Q(np.linspace(0, 1, 1000), "psi_wall").to("NUmf").magnitude
    psips = hyper.psipNU(psis)
    psips_pre = hyper_pre.psipNU(psis)
    assert np.all(np.isclose(psips, psips_pre))
