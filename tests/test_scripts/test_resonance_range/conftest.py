import gcmotion as gcm
import numpy as np
import pytest


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
