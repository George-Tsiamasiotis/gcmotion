r"""
==============================
Fixed Points XO Classification
==============================

Script/function that classifies fixed points of the GC Hamiltonian as O Points
(stable) or X Points (unstable/saddle points).
"""

import numpy as np
from collections import deque
from gcmotion.entities.profile import Profile
from gcmotion.scripts.fixed_points_bif.points_psi_to_P_theta import (
    points_psi_to_P_theta,
)
from gcmotion.scripts.utils.hessian import hessian


def XO_points_classification(
    unclassified_fixed_points: np.ndarray,
    profile: Profile,
    dtheta: float = 1e-5,
    dpsi: float = 1e-5,
    to_P_thetas: bool = False,
) -> tuple[list]:
    r"""
    Takes in an array with fixed points of the form [:math:`\theta`,
    :math:`\psi`] and , using the Hamiltonian's Hessian, it classifies them
    in X and O points, returning two deque lists for each case respectively.

    Parameters
    ----------
    unclassified_fixed_points : np.ndarray
        np.ndarray that contains point of the form [:math:`\theta_{fixed}`,
        :math:`\psi_{fixed}`].
    profile : Profile
        Profile object that contains Tokamak and Particle information.
    dtheta : float
        Finite difference parameter (very small number) used for the
        calculation of the derivatives with respect to the
        :math:`\theta` variables. Deafults to 1e-5.
    dpsi : float
        Finite difference parameter (very small number) used for
        the calculation of the derivatives with respect to the
        :math:`\psi` variables. Deafults to 1e-5.
    to_P_thetas : bool, optional
        Boolean that determines weather :math:`\psi_{fixed}` will be turned
        into :math:`P_{\theta,fixed}` in the resulting X,O Points lists.
        Defaults to ``False``.

    Returns
    -------
    Tuple containing the two lists, X_points, O_points, of the now classified
    fixed points.
    """

    # Define a Quantity object. Will be used later.
    Q = profile.Q

    # Might need regulation depending on Pzeta (close or above 0)
    if profile.PzetaNU.m >= -1e-3:
        dtheta = dpsi = 1e-9

    O_points = deque()  # Deque for stable O-points
    X_points = deque()  # Deque for unstable X-points and saddle X-points

    def WNU(theta, psi):
        # Calculate the Hamiltonian at (theta, psi)
        psi = max(
            psi, profile.psi_wallNU.m / 100
        )  # Should become small for Pzetas close to 0 because psi-->0

        W = profile.findEnergy(
            psi=Q(psi, "NUMagnetic_flux"),
            theta=theta,
            units="NUJoule",
            potential=True,
        )

        return W.m

    for fixed_point in unclassified_fixed_points:

        theta_fixed, psi_fixed = fixed_point

        Hessian = hessian(
            WNU=WNU, theta=theta_fixed, psi=psi_fixed, dtheta=dtheta, dpsi=dpsi
        )

        # Determinant of the Hessian
        det_Hessian = np.linalg.det(Hessian)

        # Classification based on determinant
        if det_Hessian < 0:
            X_points.append(fixed_point)
        elif det_Hessian > 0:
            O_points.append(fixed_point)

    if to_P_thetas:
        # We have XO points, each, of the form [thetaXO,
        # psiXO(NUMagnetic_flux)] and we will transform them to
        # [thetaXO, P_theta_XO(NUCanonical_momentum)], if asked
        X_points = points_psi_to_P_theta(X_points, profile=profile)
        O_points = points_psi_to_P_theta(O_points, profile=profile)

    return X_points, O_points
