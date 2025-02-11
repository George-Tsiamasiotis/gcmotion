import gcmotion as gcm
import gcmotion.plot as gplt
import numpy as np

# Quantity Constructor
species = "p"
div_init = gcm.DivertorNegativeInit(species)
Q = div_init.QuantityConstructor()

# Intermediate Quantities
R = div_init.R
a = div_init.a

# Construct a Tokamak
tokamak = gcm.Tokamak(
    R=R,
    a=a,
    qfactor=gcm.qfactor.DivertorNegative(),
    bfield=gcm.bfield.DivertorNegative(),
    efield=gcm.efield.Nofield(),
)

# Setup Initial Conditions
init = gcm.InitialConditions(
    species="p",
    muB=Q(1e-4, "NUMagnetic_moment"),
    Pzeta0=Q(-0.025, "NUCanonical_momentum"),
    theta0=0,
    zeta0=0,
    psi0=Q(0.3, "psi_wall"),
    t_eval=Q(np.linspace(0, 1e-4, 100000), "seconds"),
)

# Create the particle and calculate its obrit
particle = gcm.Particle(tokamak=tokamak, init=init)
event = gcm.events.when_theta(particle.theta0, 20)
particle.run(events=[event])
print(particle)

# Some Plots
gplt.qfactor_profile(particle.profile)
gplt.magnetic_profile(particle.profile, coord="rho")
gplt.particle_evolution(particle, units="NU")
gplt.particle_poloidal_drift(
    particle,
    psilim=[0, 0.5],
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
