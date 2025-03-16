import pytest
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

psi = Q(0.5, "psi_wall").to("NUmf").m
psis = Q(np.linspace(0, 1, 10000), "psi_wall").to("NUmf").m


@pytest.mark.benchmark(group="precomputed hypergeometric array")
def test_normal_hypergeometric_benchmark_array(benchmark):
    benchmark(hyper.psipNU, psis)


@pytest.mark.benchmark(group="precomputed hypergeometric array")
def test_precomputed_hypergeomteric_benchmark_array(benchmark):
    benchmark(hyper_pre.psipNU, psis)


@pytest.mark.benchmark(group="precomputed hypergeometric float")
def test_normal_hypergeometric_benchmark_float(benchmark):
    benchmark(hyper.psipNU, psi)


@pytest.mark.benchmark(group="precomputed hypergeometric float")
def test_precomputed_hypergeomteric_benchmark_float(benchmark):
    benchmark(hyper_pre.psipNU, psi)
