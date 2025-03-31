import gcmotion as gcm
import gcmotion.plot as gplt

# Quantity Constructor
species = "p"
div_init = gcm.DTTNegativeInit(species)
Q = div_init.QuantityConstructor()

# Intermediate Quantities
R = div_init.R
a = div_init.a

# Construct a Tokamak
tokamak = gcm.Tokamak(
    R=R,
    a=a,
    qfactor=gcm.qfactor.DTTNegative(),
    bfield=gcm.bfield.DTTNegative(),
    efield=gcm.efield.Nofield(),
)

# Setup Initial Conditions
init = gcm.InitialConditions(
    species="p",
    muB=Q(1e-4, "NUMagnetic_moment"),
    Pzeta0=Q(-0.05, "NUCanonical_momentum"),
    theta0=0,
    zeta0=0,
    psi0=Q(0.05411, "NUMagnetic_flux"),
)

# Create the particle and calculate its obrit
particle = gcm.Particle(tokamak=tokamak, init=init)
particle.run(method="NPeriods", stop_after=4, info=True)

# Some Plots
gplt.qfactor_profile(particle.profile)
gplt.machine_coords_profile(
    entity=particle.profile,
    which="b",
    parametric_density=250,
    mode="filled",
    flux_units="Tesla * m^2",
    E_units="keV",
    B_units="Tesla",
    I_units="NUpc",
    g_units="NUpc",
)
gplt.particle_evolution(particle, units="NU")
gplt.particle_poloidal_drift(
    particle,
    psilim=[0.5, 1],
    flux_units="NUMagnetic_flux",
    canmon_units="NUCanonical_momentum",
    E_units="keV",
)
gplt.particle_poloidal_drift(
    particle,
    psilim=[0, 1],
    flux_units="NUMagnetic_flux",
    canmon_units="NUCanonical_momentum",
    E_units="keV",
    projection="polar",
)
