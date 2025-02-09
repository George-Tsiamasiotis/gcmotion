r""" Function that calculates the fixed points and the number of fixed points for multiple
particles in a Collection form :py:class:`~gcmotion.classes.collection.Collection`, where 
each particle has a different :math:`P_{\zeta0}`.

"""

import numpy as np
from collections import deque

from gcmotion.utils.XO_points_classification import XO_points_classification as xoc
from gcmotion.scripts.fixed_points import fixed_points
from gcmotion.utils.points_psi_to_P_theta import points_psi_to_P_theta


def bifurcation(
    profiles: list | deque,
    theta_lim: list = [-np.pi, np.pi],
    psi_lim: list = [0.01, 1.3],
    method: str = "fsolve",
    dist_tol: float = 1e-3,
    fp_ic_scan_tol: float = 5 * 1e-8,
    ic_theta_grid_density: int = 400,
    ic_psi_grid_density: int = 400,
    random_fp_init_cond: bool = False,
    fp_info: bool = False,
    bif_info: bool = False,
    fp_ic_info: bool = False,
    fp_only_confined: bool = False,
    calc_energies: bool = False,
    energy_units: str = "NUJoule",
    LAR_thetas: bool = False,
):
    r"""
    Function that calculates all the fixed points of the GC Hamiltonian for multiple profiles
    with different :math:`\P_{\zeta}`'s and returns all the information in lists. Most of its
    arguments will be passed into :py:func:`fixed_points`.

        Parameters
        ----------
        profiles : list, deque
            List of profile objects that contain Tokamak and Particle information.
        theta_lim : list, optional
            Limits of the of the :math:`\theta`, :math:`\psi` search area with respect
            to the :math:`\theta` variable. Defaults to [-:math:`\pi`, :math:`\pi`].
        psi_lim : list, optional
            Limits of the of the :math:`\theta`, :math:`\psi` search area with respect
            to the :math:`\psi` variable. Defaults to [0.01 , 1.8]. CUTION: The limits are given
            normalized to :math:`\psi_{wall}`.
        method : str, optional
            String that indicates which method will be used to find the systems fixed
            points in :py:func:`single_fixed_point`. Can either be "fsolve" (deterministic)
            or "differential evolution" (stochastic). Defaults to "fsolve".
        dist_tol : float, optional
            Tolerance below which two fixed points are not considered distinct. The differences between
            both :math:`\theta` and :math:`\psi` of the fixed points must be below this tolerance for
            the fixed points to be considered the same. Defaults to 1e-3.
        fp_ic_scan_tol : float, optional
            Tolerance below which the sum of the squares of the time derivatives of the
            :math:`\theta` and :math:`\psi` variavles is considered zero. It is passed into
            :py:func:`fp_ic_scan`. Defaults to 5 * 1e-8.
        ic_theta_grid_density : int, optional
            Density of the :math:`\theta`, :math:`\psi` 2D grid to be scanned for initial conditiond
            (fixed points candidates) with respect to the :math:`\theta` variable. It is passed into
            :py:func:`fp_ic_scan` Defaults to 400.
        ic_psi_grid_density : int, optional
            Density of the :math:`\theta`, :math:`\psi` 2D grid to be scanned for initial conditiond
            (fixed points candidates) with respect to the :math:`\psi` variable. It is passed into
            :py:func:`fp_ic_scan` Defaults to 400.
        random_fp_init_cond : bool, optional
            Boolean determining weather random initial conditions are to be used instead of those
            provided by :py:func:`fp_ic_scan`. Defaults to ``False``.
        fp_info : bool, optional
            Boolean determining weather fixed points' information is to be printed. Defaults to ``False``.
        bif_info: bool, optional
            Boolean that determines weather information regarding the bifurcation process is to
            be printed. Defaults to ``False``.
        fp_ic_info : bool, optional
            Boolean determing weather information on the initial condition is to be printed.
            Defaults to ``False``.
        fp_only_confined : bool, optional
            Boolean determining if the search for :math:`\psi_{fixed}` will be conducted only for
            :math:`\psi` < :math:`\psi_{wall}` (confined particles). Defaults to ``False``.
        calc_energies : bool, optional
            Boolean determining weather the energy of each fixed point of each profile (each :math:`\P_{\zeta}`)
            is to be calculated, stored and returned. Defaults to ``False``.
        energy_units : str, optional
            String specifying the unit of the calculated fixed points' energies. Defaults to "NUJoule".
        LAR_thetas : bool, optional
            Boolean determining weather the theta values for which fixed points occur are to be
            considered known (LAR thetas are 0 and :math:`\pi`). Defaults to ``False``.

    Returns
    -------

    thetas_fixed, P_thetas_fixed, num_of_fp : tuple
        Tuple where each element is a list containing the lists of all the :math:`theta`'s
        fixed, all the :math:`P_{theta}`'s fixed and the number of fixed points found for
        each :math:`P_{\zeta}`.
    """

    first_profile = profiles[0]
    last_profile = profiles[-1]

    # Check if the partcles have different Pzeta0's
    if first_profile.PzetaNU == last_profile.PzetaNU:
        print(r"Each profile in the collection must have different $P_{\zeta0}$")
        return

    num_of_XP = deque([])
    num_of_OP = deque([])

    X_points = deque([])
    O_points = deque([])

    X_thetas = deque([])
    X_P_thetas = deque([])

    O_thetas = deque([])
    O_P_thetas = deque([])

    O_energies = deque([])
    X_energies = deque([])

    N = len(profiles)

    for idx, profile in enumerate(profiles):

        current_P_zeta = profile.PzetaNU

        current_num_of_fp, current_fp, _ = fixed_points(
            profile=profile,
            Q=profile.Q,
            method=method,
            theta_lim=theta_lim,
            psi_lim=psi_lim,
            dist_tol=dist_tol,
            fp_ic_scan_tol=fp_ic_scan_tol,
            ic_theta_grid_density=ic_theta_grid_density,
            ic_psi_grid_density=ic_psi_grid_density,
            random_init_cond=random_fp_init_cond,
            info=fp_info,
            ic_info=fp_ic_info,
            LAR_thetas=LAR_thetas,
            only_confined=fp_only_confined,
        )

        # CAUTION: The xoc function takes in psis_fixed but returns also psis_fixed
        current_X_points, current_O_points = xoc(
            unclassified_fixed_points=current_fp,
            profile=profile,
            to_P_thetas=not calc_energies,
        )

        if calc_energies:

            # Convert deque to numpy arrays for easy manipulation
            current_X_thetas, current_X_psis = (
                zip(*current_X_points) if current_X_points else ([], [])
            )
            current_O_thetas, current_O_psis = (
                zip(*current_O_points) if current_O_points else ([], [])
            )

            current_O_psis = np.array(current_O_psis)
            current_X_psis = np.array(current_X_psis)

            current_O_energies = profile.findEnergy(
                psi=profile.Q(current_O_psis, "NUMagnetic_flux"),
                theta=np.array(current_O_thetas),
                units=energy_units,
                potential=True,
            )

            current_X_energies = profile.findEnergy(
                psi=profile.Q(current_X_psis, "NUMagnetic_flux"),
                theta=np.array(current_X_thetas),
                units=energy_units,
                potential=True,
            )

            O_energies.append(current_O_energies)
            X_energies.append(current_X_energies)

            # Turn psis to P_thetas after having calculated the energy
            current_X_points = points_psi_to_P_theta(current_X_points, profile=profile)

            current_O_points = points_psi_to_P_theta(current_O_points, profile=profile)

        # Convert deque to numpy arrays for easy manipulation
        current_X_thetas, current_X_P_thetas = (
            zip(*current_X_points) if current_X_points else ([], [])
        )
        current_O_thetas, current_O_P_thetas = (
            zip(*current_O_points) if current_O_points else ([], [])
        )

        if bif_info:
            print(
                f"\nCurrent Step: {idx+1}/{N} ({100*(idx+1)/N:.1f}%) at P_z = {current_P_zeta} with {current_num_of_fp} fixed points"
            )
            print(f"Current Fixed Points: {current_fp}\n")
            print(
                f"Current X Points: {[[float(thetaX),float(P_thetaX)] for thetaX,P_thetaX in current_X_points]}\n"
            )
            print(
                f"Current O Points: {[[float(thetaO),float(P_thetaO)] for thetaO,P_thetaO in current_O_points]}\n"
            )

        num_of_XP.append(len(current_X_points))
        num_of_OP.append(len(current_O_points))

        X_points.append(current_X_points)
        O_points.append(current_O_points)

        X_thetas.append(current_X_thetas)
        X_P_thetas.append(current_X_P_thetas)

        O_thetas.append(current_O_thetas)
        O_P_thetas.append(current_O_P_thetas)

    return (
        X_thetas,
        X_P_thetas,
        O_thetas,
        O_P_thetas,
        num_of_XP,
        num_of_OP,
        X_energies,
        O_energies,
    )
