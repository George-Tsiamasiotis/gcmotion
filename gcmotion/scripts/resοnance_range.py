r"""Script that calculates the frequency at an O point (center ) for
multiple :math:`P_{\zeta}`'s or :math:`\mu`'s"""

from tqdm import tqdm
import numpy as np
from collections import deque
from gcmotion.utils.logger_setup import logger
from gcmotion.entities.profile import Profile
from gcmotion.utils.hessian import hessian

from gcmotion.configuration.scripts_configuration import ResRangeConfig

from gcmotion.scripts.fixed_points_bif.fixed_points import fixed_points
from gcmotion.scripts.fixed_points_bif.XO_points_classification import (
    XO_points_classification as xoc,
)


def omegas_max(
    profile: Profile,
    COM_values: list | deque | np.ndarray,
    **kwargs,
) -> deque:
    r"""
    Function that calculates :math:`\omega_{\theta}`s at a O Points for multiple
    :math:`P_{\zeta}`'s or :math:`\mu`'s.

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
        which_COM : str, optional
            Determines with regard to which COM (:math:`\mu` or :math:`P_{zeta}`) will the bifurcation
            analysis take place.

        Returns
        -------
        List, array, deque of all the omegas at the O points for the different :math:`P_{\zeta}`'s or :math:`\mu`'s
    """
    # Unpack parameters
    config = ResRangeConfig()
    for key, value in kwargs.items():
        setattr(config, key, value)

    N = len(COM_values)
    selected_COMNU_str = config.which_COM + "NU"

    if config.which_COM == "Pzeta":
        COM_NU_units = "NUcanmom"
    elif config.which_COM == "mu":
        COM_NU_units = "NUmagnetic_moment"

    Omegas_max = deque()

    for COM_valueNU in tqdm(COM_values, desc="Processing"):

        current_omegas_max = deque()

        setattr(profile, selected_COMNU_str, profile.Q(COM_valueNU, COM_NU_units))

        current_COMNU = COM_valueNU

        _, current_fp, _ = fixed_points(profile=profile, **kwargs)

        logger.info(
            f"Calculated fixed points ['NUMagnetic_flux'] for res_range script with {selected_COMNU_str}={current_COMNU}"
        )

        _, current_O_points = xoc(
            unclassified_fixed_points=current_fp,
            profile=profile,
            to_P_thetas=False,
        )

        logger.info(
            f"Classified fixed points ['NUmf'] for bifurcation script with {selected_COMNU_str}={current_COMNU}"
        )

        for O_Point in current_O_points:
            omega_maxNU = _omega_maxNU(profile=profile, O_Point=O_Point, delta=config.hessian_delta)

            current_omegas_max.append(profile.Q(omega_maxNU, "NUw0").to(config.freq_units).m)

        Omegas_max.append(current_omegas_max)

        logger.info(
            f"Calculated omegas_max ['{config.freq_units}'] of O points for res_range script with {selected_COMNU_str}={current_COMNU}"
        )

    return Omegas_max


def _omega_maxNU(profile: Profile, O_Point: tuple, delta: float = 1e-5) -> float:
    r"""
    Function that calculates the frequency :math:`\omega_{\theta}` at an O Point. This
    frequency will be the maximum frequency of the family of orbits occupying this
    "island" of the phase space. Therefore, (because the frequancy is 0 at the separatrix)
    this function provides the frequancy range for the entire family of orbits
    inside this "island".

        Parameters
        ----------
        profile : Profile
            Profile object that contains Tokamak and Particle information.
        O_Points : tuple
        O Point where the frequancy will be calculated.
        Has the form (:math:`\theta`, :math:`\psi`).
        delta : float
            Finite difference parameter (very small number) used for the calculation of the
            derivatives with respect to the :math:`\theta`, :math:`\psi` variables.
            Defaults to 1e-5.


        Returns
        -------
        The frequency at the O Point.

    """
    theta_O_fixed, psi_O_fixed = O_Point

    def _WNU(theta: float, psi: float):
        # Calculate the Hamiltonian at (theta, psi)
        psi = max(
            psi, profile.psi_wallNU.m / 100
        )  # Should become small for Pzetas close to 0 because psi-->0

        W = profile.findEnergy(
            psi=profile.Q(psi, "NUMagnetic_flux"), theta=theta, units="NUJoule", potential=True
        )

        return W.m

    Hessian = hessian(WNU=_WNU, theta=theta_O_fixed, psi=psi_O_fixed, delta=delta)

    d2W_dtheta2 = Hessian[0][0]
    d2W_dpsi2 = Hessian[1][1]
    d2W_dtheta_dpsi = Hessian[0][1]
    logger.info(f"Calculated the Hessian values: {d2W_dpsi2=}, {d2W_dtheta2=}, {d2W_dtheta_dpsi=}")

    dpsi_dPtheta = _dpsi_dPtheta(profile=profile, theta=theta_O_fixed, psi=psi_O_fixed)
    logger.info(f"Calculated {dpsi_dPtheta=}")

    A = dpsi_dPtheta * np.array([[d2W_dtheta_dpsi, d2W_dpsi2], [-d2W_dtheta2, -d2W_dtheta_dpsi]])

    eigA = np.linalg.eigvals(A)
    logger.info(f"Calculated A matrix eigenvalues: {eigA=}")

    omega_max = eigA.imag[0]
    logger.info(f"Got imaginary part of eigenvalues-->{omega_max=}")

    return omega_max


def _dpsi_dPtheta(profile: Profile, theta: float, psi: float) -> float:
    """Simple function that calculates the quantiti 'dpsi_dPtheta'"""
    # Tokamak profile
    qfactor = profile.qfactor
    bfield = profile.bfield

    # Define quantites for the solver for clarity
    solverqNU = qfactor.solverqNU
    psipNU = qfactor.psipNU
    solverbNU = bfield.solverbNU

    # Parameters
    Pzeta = profile.PzetaNU.m

    # Object methods calls
    q = solverqNU(psi)
    psi_p = psipNU(psi)
    _, _, currents, currents_der = solverbNU(psi, theta)

    # Unpack
    i, g = currents
    i_der, g_der = currents_der
    # Multiply current derivatives by q to get their derivatives with
    # respect to psi instead of psip
    i_der, g_der = q * i_der, q * g_der

    term1 = 1 + i / (q * g)
    term2 = i_der * (Pzeta + psi_p) / g
    term3 = -g_der / g**2 * (Pzeta + psi_p) / g * i

    dPtheta_dpsi = term1 + term2 + term3

    return 1 / dPtheta_dpsi
