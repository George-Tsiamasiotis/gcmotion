r"""
Script that finds fixed points candidates to be passed into the :py:func:`single_fixed_point`
function's numerical solver as initial conditions (fp_is_scan --> fixed points initial conditions scan)
"""

import numpy as np

from gcmotion.entities.profile import Profile
from gcmotion.scripts.fixed_points_bif.fp_system import system
from numba import njit


@njit("int64[:,:](float64[:,:])")
def _find_local_minima(arr: np.ndarray):
    r"""
    Finds local minima in a 2D array.

        Parameters
        ----------
        arr : np.ndarray
            Array whose local minima are to be located.

        Returns
        -------
        indices : list

    """

    nx, ny = arr.shape
    indices = []
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            if (
                (arr[i, j] < arr[i + 1, j + 1])
                and (arr[i, j] < arr[i + 1, j])
                and (arr[i, j] < arr[i + 1, j - 1])
                and (arr[i, j] < arr[i - 1, j + 1])
                and (arr[i, j] < arr[i - 1, j])
                and (arr[i, j] < arr[i - 1, j - 1])
                and (arr[i, j] < arr[i, j + 1])
                and (arr[i, j] < arr[i, j - 1])
            ):

                indices.append((int(i), int(j)))

    return np.array(indices)


def fp_ic_scan(
    profile: Profile,
    method: str = "fsolve",
    theta_grid_density: int = 500,
    psi_grid_density: int = 101,
    theta_lim: list = [np.pi, np.pi],
    psi_lim: list = [0.01, 1.8],
    tol: float = 1e-8,
    psi_dot_scaling_factor: float = 70,
) -> list:
    r"""
    Function that finds fixed points candidates to used as initial conditions for the
    numerical solver in :py:func:`single_fixed_point`, by scanning a :math:`\theta`, :math:`\psi`
    2D grid and storing the values (:math:`\theta`,:math:`\psi`) for which the time derivatives of
    :math:`\theta` and :math:`\psi` are local minima and (practically) zero.

        Parameters
        ----------
        profile : Profile
            Profile object that contains Tokamak and Particle information.
        method : str, optional
            String that indicates which method will be used to find the systems fixed
            points in :py:func:`single_fixed_point`. Can either be "fsolve" (deterministic)
            or "differential evolution" (stochastic). Defaults to "fsolve".
        theta_grid_density : int, optional
            Density of the :math:`\theta`, :math:`\psi` 2D grid with respect to the :math:`\theta`
            variable. Defaults to 500.
        psi_grid_density : int, optional
            Density of the :math:`\theta`, :math:`\psi` 2D grid with respect to the :math:`\psi`
            variable. Defaults to 101.
        theta_lim : list, optional
            Limits of the of the :math:`\theta`, :math:`\psi` 2D grid with respect to the :math:`\theta`
            variable. Defaults to [-:math:`\pi`, :math:`\pi`].
        psi_lim : list, optional
            Limits of the of the :math:`\theta`, :math:`\psi` 2D grid with respect to the :math:`\psi`
            variable. Defaults to [0.01 , 1.8]. CUTION: The limits are given normalized to :math:`\psi_{wall}`.
        tol : float, optional
            Tolerance that determines weather the time derivatives of the :math:`\theta` and :math:`\psi`
            variables can be considered zero. Defaults to 1e-8.
        psi_dot_scaling_factor : float,optional
            Scaling factor that is used in the sum of squares of the time derivatives of the
            :math:`\theta` and :math:`\psi` values like so -->
            :math:`\dot{\theta}^2` + (psi_dot_scaling_factor:math:`\dot{\psi})^2` because physiacally
            the time derivative of :math:`\psi` is quite smaller than that of :math:`\theta`. Defaults to 70.


        Returns
        -------
        list
            List of all the fixed points candidates fould after the 2D grid scan. They are to be
            passed in the numerical solver as initial conditions to find all the true fixed points.

    """

    # Shift the theta scanning region to the left (padding the end) in order to scan
    # the entire area of interest without finding duplicate (essentially
    # the same) candidates at -π and π
    theta_shift = abs(np.diff(theta_lim) / theta_grid_density)

    theta_min, theta_max = theta_lim[0] + theta_shift, theta_lim[1] + theta_shift
    psi_min, psi_max = psi_lim[0], psi_lim[1]

    theta_grid, psi_grid = np.meshgrid(
        np.linspace(theta_min, theta_max, theta_grid_density),
        np.linspace(psi_min, psi_max, psi_grid_density),
    )

    system_values = np.empty_like(theta_grid)
    grid_shape = theta_grid.shape

    # Precompute system values
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            if method == "differential evolution":
                system_values[i, j] = system(
                    theta_grid[i, j],
                    psi_grid[i, j],
                    profile,
                    method=method,
                    psi_dot_scaling_factor=psi_dot_scaling_factor,
                )
            elif method == "fsolve":
                theta_dot, psi_dot = system(
                    theta_grid[i, j], psi_grid[i, j], profile, method=method
                )
                system_values[i, j] = (
                    theta_dot**2 + (psi_dot_scaling_factor * psi_dot) ** 2
                )  # 60, 70, 80 worked (between 60 - 120 in general) for LAR, SMART PT
                # 30 for SMART NT

    # Find local minima
    indices = _find_local_minima(system_values)

    filtered_minima = [(i, j) for (i, j) in indices if system_values[i, j] < tol]

    # Extract minima positions
    fixed_point_candidates = [(theta_grid[i, j], psi_grid[i, j]) for i, j in filtered_minima]

    return fixed_point_candidates
