import numpy as np
import gcmotion as gcm


# Quantity Constructor
Rnum = 1.65
anum = 0.5
B0num = 1
species = "p"
Q = gcm.QuantityConstructor(R=Rnum, a=anum, B0=B0num, species=species)

# Intermediate Quantities
R = Q(Rnum, "meters")
a = Q(anum, "meters")
B0 = Q(B0num, "Tesla")
i = Q(0, "NUPlasma_current")
g = Q(1, "NUPlasma_current")

hyper = gcm.qfactor.Hypergeometric(a, B0, q0=1.1, q_wall=3.8, n=2)
hyper_pre = gcm.qfactor.PrecomputedHypergeometric(
    a, B0, q0=1.1, q_wall=3.8, n=2
)

psis = Q(np.linspace(0, 1, 10000), "psi_wall").to("NUmf").m


def test_normal_hypergeometric_benchmark(benchmark):
    benchmark(hyper.psipNU, psis)


def test_precomputed_hypergeomteric_benchmark(benchmark):
    benchmark(hyper_pre.psipNU, psis)


def test_normal_precomputed_hypergeometric_equality():
    assert np.all(np.isclose(hyper.psipNU(psis), hyper_pre.psipNU(psis)))
