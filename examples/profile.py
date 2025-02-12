import gcmotion as gcm
import gcmotion.plot as gplt

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
    qfactor=gcm.qfactor.Hypergeometric(a, B0, q0=1.1, q_wall=3.8, n=2),
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

print(profile)

# Some Plots
gplt.qfactor_profile(profile)
gplt.profile_energy_contour(profile)
gplt.profile_energy_contour(profile, projection="polar")
