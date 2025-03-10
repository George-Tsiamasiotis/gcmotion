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


def test_tokamak_instantiation():
    r"""Tests that a simple tokamak can be created without errors."""
    _ = gcm.Tokamak(
        R=R,
        a=a,
        qfactor=gcm.qfactor.Unity(),
        bfield=gcm.bfield.LAR(B0=B0, i=i, g=g),
        efield=gcm.efield.Nofield(),
    )
