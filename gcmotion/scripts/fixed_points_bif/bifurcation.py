r"""Function that calculates the fixed points, the number of fixed points, and their energies
for multiple profiles , where each profile has a different Pzeta or mu.
"""

from tqdm import tqdm
import numpy as np
from collections import deque
from gcmotion.utils.logger_setup import logger
from gcmotion.entities.profile import Profile

from gcmotion.scripts.fixed_points_bif.XO_points_classification import (
    XO_points_classification as xoc,
)
from gcmotion.scripts.fixed_points_bif.fixed_points import fixed_points
from gcmotion.utils.points_psi_to_P_theta import points_psi_to_P_theta
from gcmotion.configuration.scripts_configuration import BifurcationConfig


def bifurcation(profile: Profile, COM_values: list | deque, **kwargs) -> dict:
    r"""
    Function that calculates all the fixed points of the GC Hamiltonian for multiple profiles
    with different :math:`P_{\zeta}`'s or :math:`\mu`'s and returns all the information in lists.
    Most of its arguments will be passed into :py:func:`fixed_points`.

        Parameters
        ----------
        profile : Profile
            Profile object containing Tokamak information.
        COM_values : list, deque
            List of COM values :math:`P_{\zeta}`'s or :math:`\mu`'s in [NU].
        Other Parameters
        ----------
        thetalim : list, optional
            Limits of the of the :math:`\theta`, :math:`\psi` search area with respect
            to the :math:`\theta` variable. Defaults to [-:math:`\pi`, :math:`\pi`].
        psilim : list, optional
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
            Boolean determining weather fixed points' information is to be is to be printed in the log. Defaults to ``False``.
        bif_info: bool, optional
            Boolean that determines weather information regarding the bifurcation process is to
            be is to be printed in the log. Defaults to ``False``.
        fp_ic_info : bool, optional
            Boolean determing weather information on the initial condition is to be is to be printed in the log.
            Defaults to ``False``.
        fp_only_confined : bool, optional
            Boolean determining if the search for :math:`\psi_{fixed}` will be conducted only for
            :math:`\psi` < :math:`\psi_{wall}` (confined particles). Defaults to ``False``.
        calc_energies : bool, optional
            Boolean determining weather the energy of each fixed point of each profile (each :math:`\P_{\zeta}`)
            is to be calculated, stored and returned. Defaults to ``False``.
        energy_units : str, optional
            String specifying the unit of the calculated fixed points' energies. Defaults to "NUJoule".
        energies_info : bool, optional
            Boolean determining weather information on the fixed points' energies is to be is to be printed in the log.
            Defaults to ``True``.
        which_COM : str, optional
            Determines with regard to which COM (:math:`\mu` or :math:`P_{zeta}`) will the bifurcation
            analysis take place.

        Returns
        -------
        dict
            Dict where each value is a list containing the lists of all the :math:`theta`'s
            fixed, all the :math:`P_{theta}`'s fixed and the number of fixed points found for
            each :math:`P_{\zeta}`.
    """

    # Unpack parameters
    config = BifurcationConfig()
    for key, value in kwargs.items():
        setattr(config, key, value)

    first_COM = COM_values[0]
    last_COM = COM_values[-1]

    # Check if the partcles have different Pzeta0's
    if first_COM == last_COM and config.which_COM == "Pzeta":
        print(f"\n\nEach profile in the list must have different Pzeta\n\n")
        return
    elif first_COM == last_COM and config.which_COM == "mu":
        print(f"\n\nEach profile in the list must have different mu\n\n")
        return
    elif config.which_COM not in ["mu", "Pzeta"]:
        print("\n\n'which_COM' argumet must either be 'mu' or 'Pzeta'.\n\n")
        return

    num_of_XP = deque()
    num_of_OP = deque()

    X_points = deque()
    O_points = deque()

    X_thetas = deque()
    X_P_thetas = deque()

    O_thetas = deque()
    O_P_thetas = deque()

    O_energies = deque()
    X_energies = deque()

    N = len(COM_values)
    selected_COMNU_str = config.which_COM + "NU"

    if config.which_COM == "Pzeta":
        COM_NU_units = "NUcanmom"
    elif config.which_COM == "mu":
        COM_NU_units = "NUmagnetic_moment"

    for idx, COM_valueNU in enumerate(tqdm(COM_values, desc="Processing")):

        setattr(profile, selected_COMNU_str, profile.Q(COM_valueNU, COM_NU_units))

        current_COMNU = COM_valueNU

        current_num_of_fp, current_fp, _ = fixed_points(profile=profile, **kwargs)

        logger.info(
            f"Calculated fixed points ['NUMagnetic_flux'] for bifurcation script with {selected_COMNU_str}={current_COMNU}"
        )

        # CAUTION: The xoc function takes in psis_fixed but can return P_thetas_fixed
        # if asked
        current_X_points, current_O_points = xoc(
            unclassified_fixed_points=current_fp,
            profile=profile,
            to_P_thetas=not config.calc_energies,
        )
        logger.info(
            f"Classified fixed points ['NUCanonical_momentum'] for bifurcation script with {selected_COMNU_str}={current_COMNU}"
        )

        if config.calc_energies:

            # Unpack points
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
                units=config.energy_units,
                potential=True,
            )

            current_X_energies = profile.findEnergy(
                psi=profile.Q(current_X_psis, "NUMagnetic_flux"),
                theta=np.array(current_X_thetas),
                units=config.energy_units,
                potential=True,
            )

            O_energies.append(current_O_energies)
            X_energies.append(current_X_energies)

            if config.energies_info:
                logger.info(f"X energies {X_energies}\n\n")
                logger.info(f"O energies {O_energies}\n\n")

            logger.info(
                f"Calculated energies ['{config.energy_units}'] of fixed points for bifurcation script with {selected_COMNU_str}={current_COMNU}"
            )

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

        if config.bif_info:
            logger.info(
                f"\nCurrent Step: {idx+1}/{N} ({100*(idx+1)/N:.1f}%) at {selected_COMNU_str} = {current_COMNU} with {current_num_of_fp} fixed points"
            )
            logger.info(f"Current Fixed Points: {current_fp}\n")
            logger.info(
                f"Current X Points: {[[float(thetaX),float(P_thetaX)] for thetaX,P_thetaX in current_X_points]}\n"
            )
            logger.info(
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

    return {
        "X_thetas": X_thetas,
        "X_P_thetas": X_P_thetas,
        "O_thetas": O_thetas,
        "O_P_thetas": O_P_thetas,
        "num_of_XP": num_of_XP,
        "num_of_OP": num_of_OP,
        "X_energies": X_energies,
        "O_energies": O_energies,
    }
