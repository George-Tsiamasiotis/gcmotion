import gcmotion as gcm
import gcmotion.plot as gplt
import numpy as np

# Quantity Constructor
species = "p"
smart_init = gcm.SmartPositiveInit(species)
Q = smart_init.QuantityConstructor()

# Intermediate Quantities
R = smart_init.R
a = smart_init.a

# Construct a Tokamak
tokamak = gcm.Tokamak(
    R=R,
    a=a,
    qfactor=gcm.qfactor.SmartPositive(),
    bfield=gcm.bfield.SmartPositive(),
    efield=gcm.efield.Nofield(),
)

# Setup Initial Conditions
init = gcm.InitialConditions(
    species="p",
    muB=Q(1e-4, "NUMagnetic_moment"),
    Pzeta0=Q(-0.045, "NUCanonical_momentum"),
    theta0=0,
    zeta0=0,
    psi0=Q(0.19, "NUMagnetic_flux"),
    t_eval=Q(np.linspace(0, 2e-4, 10000), "seconds"),
)

# Create the particle and calculate its obrit
particle = gcm.Particle(tokamak=tokamak, init=init)
event = gcm.events.when_psi(init.psi0NU.m, terminal=30)
particle.run(events=[event])
print(particle)

# Some Plots
gplt.qfactor_profile(particle.profile)
gplt.machine_coords_profile(
    entity=particle.profile,
    which="b i g",
    parametric_density=250,
    mode="filled",
    flux_units="Tesla * m^2",
    E_units="keV",
    B_units="Tesla",
    I_units="NUpc",
    g_units="NUpc",
)
gplt.magnetic_profile(particle.profile, coord="rho")
gplt.particle_evolution(particle, units="NU")
gplt.particle_poloidal_drift(
    particle,
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
