r"""Simple script that calculates the Hessian matrix of the Hamiltonian at a given point
with respect ot the :math:`\theta`, :math:`\psi` variables"""

import numpy as np
from typing import Callable


def hessian(WNU: Callable, theta: float, psi: float, delta: float) -> np.ndarray:
    r"""
    Function that numerically calculates Hessian matrix of the GCM Hamiltonian at a given point
    with respect ot the :math:`\theta`, :math:`\psi` variables.

        Parameters
        ----------
        WNU : Callable
            The function that gives the value of the Hamiltonian at a certain point :math:`\theta`, :math:`\psi`.
        theta : float
            :math:`\theta` value for which second order derivatives are to be calculated.
        psi : float
            :math:`\psi` value in [NU] for which second order derivatives are to be calculated.
        delta : float
            Finite difference parameter (very small number) used for the calculation of the
            derivatives with respect to the :math:`\theta`, :math:`\psi` variables.

        Returns
        -------
        The Hessian of the GCM Hamiltonian calculated at a point [:math:`\theta`, :math:`\psi`]


    """

    # Compute the Hessian matrix elements
    d2W_dtheta2 = _higher_order_second_derivative(WNU, theta, psi, delta, delta, "x")
    d2W_dpsi2 = _higher_order_second_derivative(WNU, theta, psi, delta, delta, "y")
    d2W_dtheta_dpsi = _higher_order_second_derivative(WNU, theta, psi, delta, delta, "mixed")
    # Hessian matrix
    Hessian = np.array([[d2W_dtheta2, d2W_dtheta_dpsi], [d2W_dtheta_dpsi, d2W_dpsi2]])

    return Hessian


# Higher-order central difference for second derivatives
def _higher_order_second_derivative(
    f: Callable, x: float, y: float, dx: float, dy: float, respect_to: str
) -> float:
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
        dy : float
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
