import numpy as np
import gcmotion as gcm
import gcmotion.plot as gplt

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

# Create a Profile to see the phase space and pick Energy spans
profile = gcm.Profile(
    tokamak=tokamak,
    species=species,
    mu=Q(1.9173e-6, "NUMagnetic_moment"),
    Pzeta=Q(-0.025, "NUCanonical_momentum"),
)
gplt.fixed_points_energy_contour(
    profile,
    psilim=(0.85, 1.30),
    log_base=1 + 1e-15,
    levels=50,
    E_units="NUJoule",
    separatrices=True,
)

# =========================== Frequency Analysis ===========================

muspan = np.array([profile.muNU.m])
Pzetaspan = np.array([profile.PzetaNU.m])
Espan = np.concat(
    (
        np.linspace(5.1e-6, 1e-5, 600),
        # Zoom in to specific Energy "areas" that are close to O and X points,
        # as seen by simple contouring or bifurcation analysis
        np.linspace(6.25e-6, 6.4e-6, 100),
        np.linspace(5.65e-6, 5.9e-6, 100),
        np.linspace(7.48e-6, 7.68e-6, 100),
    )
)

freq = gcm.FrequencyAnalysis(
    tokamak=tokamak,
    psilim=(0.7, 1.4),
    muspan=muspan,
    Pzetaspan=Pzetaspan,
    Espan=Espan,
    pzeta_min_step=8e-6,  # Important! must scale with the Pζ values
    qkinetic_cutoff=9,
    main_grid_density=700,
)

freq.start()
print(freq)

df = freq.to_dataframe()
print(df)

freq.scatter(x="Energy", y="qkinetic")
freq.scatter(x="Energy", y="omega_theta")
freq.scatter(x="Energy", y="omega_zeta")
