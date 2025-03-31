import gcmotion as gcm
import numpy as np

from gcmotion.scripts.utils.hessian import hessian

# ============================TESTING FUNCTIONS==============================


def test_res_range_instantiation(simple_profile):
    r"""Tests res_range module correct instantiation essentally."""
    N = 5
    musNU = np.linspace(1e-5, 1e-4, N)

    Omega_thetas_max, Omega_zetas_max = gcm.omegas_max(
        profile=simple_profile,
        COM_values=musNU,
        which_COM="mu",
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
    )

    assert len(Omega_thetas_max) == len(Omega_zetas_max) == N


def test_martix_eigenvalues(simple_profile, bifurcation_output_no_energies):
    r"""Tests some expected properties of the eigenvalues of the matrix
    used to calculate omega thetas at O Points."""
    bifurcation_output = bifurcation_output_no_energies

    O_thetas = bifurcation_output["O_thetas"]
    O_thetas = [Otheta[0] for Otheta in O_thetas]  # necessary fromatting
    O_psis = bifurcation_output["O_psis"]

    eigenvalues = []

    def _WNU(theta: float, psi: float):
        # Calculate the Hamiltonian at (theta, psi)
        psi = max(
            psi, simple_profile.psi_wallNU.m / 100
        )  # Should become small for Pzetas close to 0 because psi-->0

        W = simple_profile.findEnergy(
            psi=simple_profile.Q(psi, "NUMagnetic_flux"),
            theta=theta,
            units="NUJoule",
            potential=True,
        )

        return W.m

    for theta_O_fixed, psi_O_fixed in zip(O_thetas, O_psis):

        Hessian = hessian(
            WNU=_WNU,
            theta=theta_O_fixed,
            psi=psi_O_fixed,
            dtheta=1e-5,
            dpsi=1e-5,
        )

        d2W_dtheta2 = Hessian[0][0]
        d2W_dpsi2 = Hessian[1][1]
        d2W_dtheta_dpsi = Hessian[0][1]

        # dpsi_dPtheta = 1 for simple profile
        A = 1 * np.array(
            [[d2W_dtheta_dpsi, d2W_dpsi2], [-d2W_dtheta2, -d2W_dtheta_dpsi]]
        )

        eigA = np.linalg.eigvals(np.squeeze(A))
        eigenvalues.append(eigA)

    assert _check_eigenvalues(eigenvalues=eigenvalues)


def _check_eigenvalues(eigenvalues, tol=1e-15):
    r"""Auxiliary function that checks for every eigenvalue element weather the
    real parts of the 2 eigenvalues are close to 0 and the imaginary parts are
    opposite.
    """
    for eig in eigenvalues:
        real_condition = np.all(np.isclose(np.real(eig), 0, atol=tol))
        imag_condition = np.isclose(np.sum(np.imag(eig)), 0, atol=tol)

        if not (real_condition and imag_condition):
            return False  # If any element fails, return False
    return True
