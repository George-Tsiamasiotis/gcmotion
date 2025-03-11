import gcmotion as gcm
import numpy as np

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
Ea = Q(73500, "Volts/meter")

# Construct a Tokamak
tokamak = gcm.Tokamak(
    R=R,
    a=a,
    qfactor=gcm.qfactor.Hypergeometric(a, B0, q0=1.1, q_wall=3.8, n=2),
    bfield=gcm.bfield.LAR(B0=B0, i=i, g=g),
    efield=gcm.efield.Radial(a, Ea, B0, peak=0.98, rw=1 / 50),
)


def test_initial_conditions_muB_keV():
    init = gcm.InitialConditions(
        species=species,
        muB=Q(0.5, "keV"),
        Pzeta0=Q(-0.025, "NUCanonical_momentum"),
        theta0=np.pi,
        zeta0=0,
        psi0=Q(0.132, "Magnetic_flux"),
        t_eval=Q(np.linspace(0, 1e-3, 100000), "seconds"),
    )
    # Make sure these work in both cases since they change after
    # _calculate_full_set() is called
    init.__repr__()
    init.__str__()

    init._calculate_full_set(tokamak)

    # Make sure these work in both cases
    init.__repr__()
    init.__str__()

    assert (
        str(init.muB.dimensionality) == "[mass] * [length] ** 2 / [time] ** 2"
    )
    assert str(init.mu.dimensionality) == "[current] * [length] ** 2"


def test_initial_conditions_muB_magnetic_moment():
    init = gcm.InitialConditions(
        species=species,
        muB=Q(1e-6, "NUMagnetic_moment"),
        Pzeta0=Q(-0.025, "NUCanonical_momentum"),
        theta0=np.pi,
        zeta0=0,
        psi0=Q(0.132, "Magnetic_flux"),
        t_eval=Q(np.linspace(0, 1e-3, 100000), "seconds"),
    )
    # Make sure these work in both cases since they change after
    # _calculate_full_set() is called
    init.__repr__()
    init.__str__()

    init._calculate_full_set(tokamak)

    # Make sure these work in both cases
    init.__repr__()
    init.__str__()

    assert (
        str(init.muB.dimensionality) == "[mass] * [length] ** 2 / [time] ** 2"
    )
    assert str(init.mu.dimensionality) == "[current] * [length] ** 2"
