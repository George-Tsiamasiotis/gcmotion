import numpy as np
import pytest
import gcmotion as gcm


@pytest.fixture(scope="module")
def giorgos_efield_profile():
    r"""Creates a profile for which the fixed points have been cross checked."""
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
        qfactor=gcm.qfactor.PrecomputedHypergeometric(
            a, B0, q0=1.0, q_wall=3.8, n=2
        ),
        bfield=gcm.bfield.LAR(B0=B0, i=i, g=g),
        efield=gcm.efield.Radial(a, Ea, B0, peak=0.98, rw=1 / 50),  # 0.98
    )

    # Create a Profile
    profile = gcm.Profile(
        tokamak=tokamak,
        species=species,
        mu=Q(1e-5, "NUMagnetic_moment"),
        Pzeta=Q(-0.0272, "NUCanonical_momentum"),
    )

    return profile


@pytest.fixture(scope="module")
def bifurcation_output_energies(simple_profile):
    r"""Runs bifurcation analysis on a simple profile."""
    N = 4

    PzetasNU = np.linspace(-0.028, -0.005, N)

    bifurcation_output = gcm.bifurcation(
        simple_profile,
        COM_values=PzetasNU,
        thetalim=[-np.pi, np.pi],
        psilim=[
            0,
            1.5,
        ],
        fp_method="fsolve",
        dist_tol=5e-4,
        fp_ic_scan_tol=1e-8,
        ic_fp_theta_grid_density=101,
        ic_fp_psi_grid_density=500,
        fp_ic_scaling_factor=90,
        calc_energies=True,
    )

    return bifurcation_output, N


@pytest.fixture(scope="session")
def bifurcation_output_no_energies(simple_profile):
    r"""Runs bifurcation analysis on a simple profile without
    calculating energies."""
    N = 5

    PzetasNU = np.linspace(-0.028, -0.005, N)

    bifurcation_output = gcm.bifurcation(
        simple_profile,
        COM_values=PzetasNU,
        thetalim=[-np.pi, np.pi],
        psilim=[
            0,
            1.5,
        ],
        fp_method="fsolve",
        dist_tol=5e-4,
        fp_ic_scan_tol=1e-8,
        ic_fp_theta_grid_density=101,
        ic_fp_psi_grid_density=500,
        fp_ic_scaling_factor=90,
        calc_energies=False,
    )

    return bifurcation_output


@pytest.fixture(scope="module")
def fp_output(giorgos_efield_profile):
    r"""Returns the fixed points for the profile created in this
    folder's conftest (cross checked efield)."""
    fixed_points_output = gcm.fixed_points(
        profile=giorgos_efield_profile,
        thetalim=[-np.pi, np.pi],
        psilim=[
            0,
            1.5,
        ],
        flux_units="psi_wall",
        fp_method="fsolve",
        dist_tol=5e-4,
        fp_ic_scan_tol=1e-8,
        ic_fp_theta_grid_density=101,
        ic_fp_psi_grid_density=500,
        fp_ic_scaling_factor=90,
    )

    return fixed_points_output
