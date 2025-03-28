import numpy as np
import pytest
import gcmotion as gcm

from gcmotion.scripts.parabolas import calc_parabolas_tpb


@pytest.fixture(scope="function")
def events(simple_particle, request, terminal):
    r"""All availiable events with simple_particle's parameters."""
    theta = simple_particle.theta0
    psi = simple_particle.psi0NU
    zeta = simple_particle.zeta0
    rho = simple_particle.rho0

    if request.param == "when_theta":
        return gcm.events.when_theta(root=theta, terminal=terminal)
    if request.param == "when_psi":
        return gcm.events.when_psi(root=psi, terminal=terminal)
    if request.param == "when_zeta":
        return gcm.events.when_zeta(root=zeta, terminal=terminal)
    if request.param == "when_rho":
        return gcm.events.when_rho(root=rho, terminal=terminal)


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
def parabolas_output(simple_profile):
    r"""Calculates tha values of the parabolas and the TP-boundary
    for a simple profile."""
    parabolas_output = calc_parabolas_tpb(
        profile=simple_profile, calc_TPB=True
    )

    return parabolas_output
