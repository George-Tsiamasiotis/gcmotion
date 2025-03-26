import gcmotion as gcm
import gcmotion.plot as gplt
import numpy as np

# ========================== QUANTITY CONSTRUCTOR ==========================

Rnum = 1.65
anum = 0.5
B0num = 1
species = "d"
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
    qfactor=gcm.qfactor.PrecomputedHypergeometric(
        a, B0, q0=1.1, q_wall=3.8, n=2
    ),
    bfield=gcm.bfield.LAR(B0=B0, i=i, g=g),
    efield=gcm.efield.Nofield(),
)

# Create a Profile
profile = gcm.Profile(
    tokamak=tokamak,
    species=species,
    mu=Q(1e-5, "NUMagnetic_moment"),
    Pzeta=Q(-0.015, "NUCanonical_momentum"),
)

N = 100
PzetaNUmin = -0.028
PzetaNUmax = -0.005
PzetasNU = np.linspace(PzetaNUmin, PzetaNUmax, N)

# Draw fixed point on energy profile contour
gplt.fixed_points_energy_contour(
    profile,
    thetalim=[-np.pi, np.pi],
    psilim=[
        0,
        1,
    ],
    # flux_units="NUmf",
    fp_ic_scan_tol=1e-8,
    ic_fp_theta_grid_density=101,
    ic_fp_psi_grid_density=500,
    fp_ic_scaling_factor=90,
)

# Draw fixed point bifurcation diagram with respect to Pzeta
gplt.bifurcation_plot(
    profile=profile,
    COM_values=PzetasNU,
    thetalim=[-np.pi, np.pi],
    psilim=[
        0,
        1.5,
    ],
    fp_ic_scan_tol=1e-8,
    ic_fp_theta_grid_density=101,
    ic_fp_psi_grid_density=500,
    fp_ic_scaling_factor=90,
    flux_units="psi_wall",
    which_COM="Pzeta",
)
