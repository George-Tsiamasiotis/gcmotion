r"""
Script/function that classifies fixed points of the GC Hamiltonian as O Points (stable)
or X Points (unstable/saddle points).
"""

import numpy as np
from collections import deque
from gcmotion.entities.profile import Profile
from gcmotion.utils.points_psi_to_P_theta import points_psi_to_P_theta
from typing import Callable


# Higher-order central difference for second derivatives
def _higher_order_second_derivative(
    f: Callable, x: float, y: float, dx: float, dy: float, respect_to: str
):
    r"""
    Function that numerically calculates the second derivatives (including mixed) of a two
    variable function.

        Parameters
        ----------
        f : Callable
            Two variable function, whose second order derivatives are to be calculated.
        x : float
            x value for which second order derivatives are to be calculated.
        y : float
            y value for which second order derivatives are to be calculated.
        dx : float
            Finite difference parameter (very small number) used for the calculation of the
            derivatives with respect to the x variable.
        dxy : float
            Finite difference parameter (very small number) used for the calculation of the
            derivatives with respect to the y variable.
        respect_to : str
            String that can be either "x" or "y" and indicates with respect to which
            variable (x or y) will the derivatives be calculated.



        Returns
        -------
        Second order regular or mixed derivative of function f at the selected point (x , y).


    """
    if respect_to == "x":
        return (
            -f(x + 2 * dx, y)
            + 16 * f(x + dx, y)
            - 30 * f(x, y)
            + 16 * f(x - dx, y)
            - f(x - 2 * dx, y)
        ) / (12 * dx**2)
    elif respect_to == "y":
        return (
            -f(x, y + 2 * dy)
            + 16 * f(x, y + dy)
            - 30 * f(x, y)
            + 16 * f(x, y - dy)
            - f(x, y - 2 * dy)
        ) / (12 * dy**2)
    else:  # Mixed derivative
        return (f(x + dx, y + dy) - f(x + dx, y - dy) - f(x - dx, y + dy) + f(x - dx, y - dy)) / (
            4 * dx * dy
        )


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
        d2W_dtheta2 = _higher_order_second_derivative(
            WNU, theta_fixed, psi_fixed, delta, delta, "x"
        )
        d2W_dpsi2 = _higher_order_second_derivative(WNU, theta_fixed, psi_fixed, delta, delta, "y")
        d2W_dtheta_dpsi = _higher_order_second_derivative(
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
        # We have XO points, each, of the form [thetaXO, psiXO(NUMagnetic_flux)] and
        # we will transform them to [thetaXO, P_theta_XO(NUCanonical_momentum)], if asked
        X_points = points_psi_to_P_theta(X_points, profile=profile)
        O_points = points_psi_to_P_theta(O_points, profile=profile)

    return X_points, O_points
