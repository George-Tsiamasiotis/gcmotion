import gcmotion as gcm
import gcmotion.plot as gplt

# ========================== QUANTITY CONSTRUCTOR ==========================

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

# Construct a Tokamak
tokamak = gcm.Tokamak(
    R=R,
    a=a,
    qfactor=gcm.qfactor.Hypergeometric(a, B0, q0=1.1, q_wall=3.8, n=2),
    bfield=gcm.bfield.LAR(B0=B0, i=i, g=g),
    efield=gcm.efield.Nofield(),
)

# Create a Profile
profile = gcm.Profile(
    tokamak=tokamak,
    species=species,
    mu=Q(1e-5, "NUMagnetic_moment"),
    Pzeta=Q(-0.028, "NUCanonical_momentum"),
)

gplt.parabolas_diagram(
    profile=profile,
    Pzetalim=[-1.45, 0.35],  # Given as PzetaNU/psip_wallNU
    enlim=[0.25, 3],  # Given as ENU/muB0NU
    Pzeta_density=1000,
    plot_TPB=True,
    show_d_line=True,
)
