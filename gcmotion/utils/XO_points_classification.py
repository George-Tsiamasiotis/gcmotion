r"""
Script/function that classifies fixed points of the GC Hamiltonian as O Points (stable)
or X Points (unstable/saddle points).
"""

import numpy as np
from collections import deque
from gcmotion.entities.profile import Profile
from gcmotion.utils.second_derivative import higher_order_second_derivative
from gcmotion.utils.points_psi_to_P_theta import points_psi_to_P_theta


def XO_points_classification(
    unclassified_fixed_points: np.ndarray,
    profile: Profile,
    delta: float = 1e-5,
    to_P_thetas: bool = False,
):
    r"""
    Takes in an array with fixed points of the form [:math:`\theta`, :math:`\psi`] and
    , using the Hamiltonian's Hessian, it classifies them in X and O points, returning two
    deque lists for each case respectively.

        Parameters
        ----------
        unclassified_fixed_points : np.ndarray
            np.ndarray that contains point of the form [:math:`\theta_{fixed}`, :math:`\psi_{fixed}`].
        profile : Profile
            Profile object that contains Tokamak and Particle information.
        delta : float, optional
            Very small number used to calculate the second order derivatives, with
            a finite difference method, needed for the Hessian. Deafults to 1e-5.
        to_P_thetas : bool, optional
            Boolean that determines weather :math:`\psi_{fixed}` will be turned into
            :math:`P_{\theta,fixed}` in the resulting X,O Points lists. Defaults to ``False``.



        Returns
        -------

        X_points, O_points : tuple
            Tuple containing the two lists, X_points, O_points, of the now classified
            fixed points.


    """

    # Define a Quantity object. Will be used later.
    Q = profile.Q

    # Might need regulation depending on Pzeta (close or above 0)
    if profile.PzetaNU.m >= -1e-3:
        delta = 1e-9

    O_points = deque([])  # Deque for stable O-points
    X_points = deque([])  # Deque for unstable X-points and saddle X-points

    def WNU(theta, psi):
        # Calculate the Hamiltonian at (theta, psi)
        psi = max(1e-3, psi)
        W = profile.findEnergy(
            psi=Q(psi, "NUMagnetic_flux"), theta=theta, units="NUJoule", potential=True
        )

        return W.m

    for fixed_point in unclassified_fixed_points:

        theta_fixed, psi_fixed = fixed_point

        # Compute the Hessian matrix elements
        d2W_dtheta2 = higher_order_second_derivative(WNU, theta_fixed, psi_fixed, delta, delta, "x")
        d2W_dpsi2 = higher_order_second_derivative(WNU, theta_fixed, psi_fixed, delta, delta, "y")
        d2W_dtheta_dpsi = higher_order_second_derivative(
            WNU, theta_fixed, psi_fixed, delta, delta, "mixed"
        )
        # Hessian matrix
        Hessian = np.array([[d2W_dtheta2, d2W_dtheta_dpsi], [d2W_dtheta_dpsi, d2W_dpsi2]])

        # Determinant of the Hessian
        det_Hessian = np.linalg.det(Hessian)

        # Classification based on determinant
        if det_Hessian < 0:
            X_points.append(fixed_point)
        elif det_Hessian > 0:
            O_points.append(fixed_point)

    if to_P_thetas:
        # We have XO points, each, of the form [thetaXO, psiXO] and we will transform them
        # to [thetaXO, P_theta_XO], if asked
        X_points = points_psi_to_P_theta(X_points, profile=profile)
        O_points = points_psi_to_P_theta(O_points, profile=profile)

    return X_points, O_points
