import gcmotion as gcm
import gcmotion.plot as gplt
import numpy as np

# Quantity Constructor
species = "p"
smart_init = gcm.SmartNegativeInit(species)
Q = smart_init.QuantityConstructor()

# Intermediate Quantities
R = smart_init.R
a = smart_init.a
B0 = smart_init.B0
i = Q(0, "NUPlasma_current")
g = Q(1, "NUPlasma_current")

# Construct a Tokamak
tokamak = gcm.Tokamak(
    R=R,
    a=a,
    qfactor=gcm.qfactor.SmartNegative(),
    bfield=gcm.bfield.SmartNegative(),
    efield=gcm.efield.Nofield(),
)

# Setup Initial Conditions
init = gcm.InitialConditions(
    species="p",
    muB=Q(1e-3, "NUMagnetic_moment"),
    Pzeta0=Q(-0.05, "NUCanonical_momentum"),
    theta0=0,
    zeta0=0,
    psi0=Q(0.076, "NUMagnetic_flux"),
    t_eval=Q(np.linspace(0, 1e-4, 10000), "seconds"),
)

# Create the particle and calculate its obrit
particle = gcm.Particle(tokamak=tokamak, init=init)
event = gcm.events.when_psi(init.psi0, terminal=15)
particle.run(events=[event])
print(particle)


# Some plots
gplt.qfactor_profile(particle.profile)
gplt.magnetic_profile(particle.profile, coord="rho")
gplt.particle_evolution(particle, units="NU")
gplt.particle_poloidal_drift(
    particle,
    psilim="auto",
    flux_units="NUMagnetic_flux",
    canmon_units="NUCanonical_momentum",
    E_units="keV",
    levels=80,
)
gplt.particle_poloidal_drift(
    particle,
    psilim="auto",
    flux_units="psi_wall",
    canmon_units="NUCanonical_momentum",
    E_units="keV",
    projection="polar",
    levels=80,
)
