import pytest
import numpy as np
import gcmotion as gcm

from gcmotion.configuration.scripts_configuration import PrecomputedConfig


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


psi_max = PrecomputedConfig.psi_max


@pytest.mark.parametrize(
    "zupper",
    [
        np.linspace(0.1, psi_max, 11).tolist(),
        pytest.param(
            [psi_max + 1, psi_max + 2],
            marks=pytest.mark.xfail(reason="Over the spline limits"),
        ),
    ],
)
def test_analytic_and_precomputed_hyp2f1(Q, hyper, hyper_pre, zupper):
    psis = Q(np.linspace(0, zupper, 100000), "psi_wall").to("NUmf").magnitude
    psips = hyper.psipNU(psis)
    psips_pre = hyper_pre.psipNU(psis)
    assert np.all(np.isclose(psips, psips_pre))
