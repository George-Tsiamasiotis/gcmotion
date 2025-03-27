import gcmotion as gcm
import gcmotion.plot as gplt
import numpy as np

# Quantity Constructor
species = "p"
div_init = gcm.DTTPositiveInit(species)
Q = div_init.QuantityConstructor()

# Intermediate Quantities
R = div_init.R
a = div_init.a
B0 = div_init.B0
i = Q(0, "NUPlasma_current")
g = Q(1, "NUPlasma_current")

# Construct a Tokamak
tokamak = gcm.Tokamak(
    R=R,
    a=a,
    qfactor=gcm.qfactor.DTTPositive(),
    bfield=gcm.bfield.DTTPositive(),
    efield=gcm.efield.Nofield(),
)

profile = gcm.Profile(
    tokamak=tokamak,
    species=species,
    mu=Q(1e-4, "NUMagnetic_moment"),
    Pzeta=Q(-0.04, "NUCanonical_momentum"),
)

N = 50
muNUmin = 1e-5
muNUmax = 1e-3
musNU = np.linspace(muNUmin, muNUmax, N)

# Draw fixed point on energy profile contour
gplt.fixed_points_energy_contour(
    profile,
    thetalim=[-np.pi, np.pi],
    psilim=[
        0,
        1,
    ],
    fp_ic_scan_tol=1e-6,
    ic_fp_theta_grid_density=101,
    ic_fp_psi_grid_density=200,
    fp_ic_scaling_factor=120,
)

# # Draw fixed point bifurcation diagram with respect to mu
gplt.bifurcation_plot(
    profile=profile,
    COM_values=musNU,
    thetalim=[-np.pi, np.pi],
    psilim=[
        0,
        1,
    ],
    fp_ic_scan_tol=1e-6,
    ic_fp_theta_grid_density=101,
    ic_fp_psi_grid_density=200,
    fp_ic_scaling_factor=120,
    which_COM="mu",
)
