"""
Script/Function that calculates the fixed points of the GC Hamiltonian for a given profile
(magnetic field, electric field, qfactor, particle species etc.)
"""

import numpy as np
import pint
from itertools import product
from gcmotion.utils.logger_setup import logger
from collections import deque

from gcmotion.entities.profile import Profile
from gcmotion.utils.fixed_points_bif.fp_ic_scan import fp_ic_scan as ic_scanner
from gcmotion.utils.fixed_points_bif.single_fixed_point import fixed_point
from gcmotion.configuration.fixed_points_bifurcation_parameters import FixedPointsConfig

# Quantity alias for type annotations
type Quantity = pint.UnitRegistry.Quantity


def fixed_points(profile: Profile, **kwargs) -> tuple[list, list, list]:
    r"""
    Function that finds the fixed points of the GC Hamiltonian by numerically setting the
    time derivatives of the :math:`\theta` and :math:`\psi` variables equal to zero (numerical
    solvers) for a number of different initial conditions. White's equations are used.

        Parameters
        ----------
        profile : Profile
            Profile object that contains Tokamak and Particle information.

        Other Parameters
        ----------
        method : str, optional
            String that indicates which method will be used to find the systems fixed
            points in :py:func:`single_fixed_point`. Can either be "fsolve" (deterministic)
            or "differential evolution" (stochastic). Defaults to "fsolve".
        dist_tol : float
            Tolerance below which two fixed points are not considered distinct. The differences between
            both :math:`\theta` and :math:`\psi` of the fixed points must be below this tolerance for
            the fixed points to be considered the same. Defaults to 1e-3.
        fp_ic_scan_tol : float
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
        psi_dot_scaling_factor : float,optional
            Scaling factor that is used in the sum of squares of the time derivatives of the
            :math:`\theta` and :math:`\psi` values like so -->
            :math:`\dot{\theta}^2` + (psi_dot_scaling_factor:math:`\dot{\psi})^2` because physiacally
            the time derivative of :math:`\psi` is quite smaller than that of :math:`\theta`. Defaults to 70.
        random_init_cond : bool, optional
            Boolean determining weather random initial conditions are to be used instead of those
            provided by :py:func:`fp_ic_scan`. Defaults to ``False``.
        info : bool, optional
            Boolean determining weather fixed points' information is to be is to be printed in the log.
            Defaults to ``False``.
        ic_info : bool, optional
            Boolean determing weather information on the initial condition is to be is to be printed in the log.
            Defaults to ``False``.
        only_confined : bool, optional
            Boolean determining if the search for :math:`\psi_{fixed}` will be conducted only for
            :math:`\psi` < :math:`\psi_{wall}` (confined particles). Defaults to ``False``.


        Returns
        -------
        tuple
            Tuple that contains the number of the distinct fixed points found, the distinct fixed points,
            as well as the initial conditions used to locate these fixed points. :math:`\psi` is calculated in
            "NUMagnetic_flux" units.

    """
    # Unpack Parameters
    config = FixedPointsConfig()
    for key, value in kwargs.items():
        setattr(config, key, value)

    # print(f"\n\nCONFIG_CLASS:{vars(config)}\n\n")

    logger.info(
        f"Converting psi_lim from {config.psilim} 'NUpsi_wall' units to 'NUMagnetic_flux units'"
    )

    psi_lim = profile.Q(config.psilim, "NUpsi_wall").to("NUmagnetic_flux").m

    logger.info(f"Converted psi_lim from 'NUpsi_wall' units to {psi_lim} 'NUMagnetic_flux units'")

    if config.fp_only_confined:
        psi_lim[1] = profile.psi_wall.to("NUMagnetic_flux").m
        logger.info(f"Set psi_max={psi_lim[1]} for only confined search")

    bounds, initial_conditions, fixed_points = _set_up_fixed_points(
        profile=profile,
        method=config.fp_method,
        thetalim=config.thetalim,
        psilim=psi_lim,
        fp_ic_scan_tol=config.fp_ic_scan_tol,
        ic_theta_grid_density=config.ic_fp_theta_grid_density,
        ic_psi_grid_density=config.ic_fp_psi_grid_density,
        random_init_cond=config.fp_random_init_cond,
        ic_scaling_factor=config.fp_ic_scaling_factor,
    )

    logger.info(f"Ran _set_up_fixed_points for psilim={psi_lim}")

    idx = 0

    # Run fixed_point() for multiple initial conditions in order to locate
    # multiple fixed points.
    for initial_condition in initial_conditions:

        theta_fix, psi_fix = fixed_point(
            method=config.fp_method,
            initial_condition=initial_condition,
            bounds=bounds,
            profile=profile,
        )
        fixed_points[idx] = (float(theta_fix), float(psi_fix))
        idx += 1

    logger.info(
        f"Calculated fixed points {fixed_points} ['NUMagnetic_flux'] in fixed_points script"
    )
    # If the fixed points that were found have identical values-->
    # find out how many distinct fixed points were located
    distinct_fixed_points = _distinctify(fixed_points, tol=config.dist_tol)
    num_of_dfp = distinct_fixed_points.shape[0]

    logger.info(
        f"Calculated {num_of_dfp} distinct fixed points {distinct_fixed_points} ['NUMagnetic_flux'] in fixed_points script"
    )

    if config.fp_ic_info:
        initial_conditions_print = [[float(x), float(y)] for x, y in initial_conditions]
        logger.info(f"\nInitial Conditions:{initial_conditions_print}\n")
        logger.info(f"\nNumber of Initial Conditions: {len(initial_conditions_print)}\n")

    if config.fp_info and fixed_points.shape[0] <= 30:
        logger.info(f'\nUsing Method: "{config.fp_method}"\n')
        logger.info(f"\nFixed Points: {fixed_points}\n")
        logger.info(f"Number of Fixed Points: {fixed_points.shape[0]}")

    if config.fp_info and distinct_fixed_points.shape[0] <= 12:
        logger.info(f"\nDistinct Fixed Points: {distinct_fixed_points}\n")
        logger.info(f"Number of Distinct Fixed Points: {distinct_fixed_points.shape[0]}\n")

    return num_of_dfp, distinct_fixed_points, initial_conditions


def _distinctify(points: np.ndarray | list | deque, tol: float):
    r"""
    Simple function that determines which elements [,] of a list of lists of len 2 [[,],[,],[,]...]
    can be considered distinct from one another. In this project's context, it is used to
    determine which points [x,y] can be considered distinct.

        Parameters
        ----------
        points : np.ndarray | list | deque
            Iterable (np.ndarray, list, deque) that contains sublists that are to be examined for uniquness.
        tol : list
            If two sublists have both elements that are less than tol (tolerance) apart, they
            are considered idenical.

        Returns
        -------
        List that contains only the distinct sublists/points.


    """

    if isinstance(points, deque):
        points = np.array(points)

    def are_considered_equal(sublist1, sublist2, tol=tol):
        return abs(sublist1[0] - sublist2[0]) <= tol and abs(sublist1[1] - sublist2[1]) <= tol

    distinct_points = np.empty((points.shape[0], points.shape[1]))
    distinct_points[:] = np.nan

    idx = 0

    for point in points:

        is_unique = True
        for distinct in distinct_points:
            if are_considered_equal(distinct, point):
                is_unique = False
                break

        if is_unique:
            distinct_points[idx] = point
            idx += 1

    distinct_points = distinct_points[~np.isnan(distinct_points).any(axis=1)]

    return distinct_points


def _set_up_fixed_points(
    profile: Profile,
    method: str,
    thetalim: list,
    psilim: list,
    fp_ic_scan_tol: float,
    ic_theta_grid_density: int,
    ic_psi_grid_density: int,
    ic_scaling_factor: int,
    random_init_cond: bool = False,
) -> tuple:
    """
    Function that sets up some parameters of the fixed points' system, numerical solvers,
    initial conditions.
    """

    # CAUTION: Here psi lim has already been denormalized from psi_wall and converted to
    # NUMagnetic_flux. This happened in fixed_points()
    psi_min = psilim[0]
    psi_max = psilim[1]

    theta_min = thetalim[0]
    theta_max = thetalim[1]

    if random_init_cond:

        bounds = [(theta_min, theta_max), (0.99 * psi_min, 1.01 * psi_max)]

        theta_ic = np.linspace(theta_min, theta_max, ic_theta_grid_density)
        psi_ic = np.linspace(psi_min, psi_max, ic_psi_grid_density)

        initial_conditions = [
            [theta_init, psi_init] for theta_init, psi_init in product(theta_ic, psi_ic)
        ]

        empty_fixed_points = np.empty((len(initial_conditions), len(initial_conditions[0])))
        empty_fixed_points[:] = np.nan

    else:

        bounds = [(theta_min, theta_max), (0.99 * psi_min, 1.01 * psi_max)]

        initial_conditions = ic_scanner(
            profile=profile,
            method=method,
            theta_grid_density=ic_theta_grid_density,
            psi_grid_density=ic_psi_grid_density,
            psi_lim=psilim,
            theta_lim=thetalim,
            tol=fp_ic_scan_tol,
            psi_dot_scaling_factor=ic_scaling_factor,
        )

        # Make sure there are initial conditions. If not, use randoom
        if not initial_conditions:
            logger.info(f"No initial conditions were found from fp_ic_scan")

            print("\n\nDID NOT FIND INITIAL CONDITIONS. USING RANDOM...\n\n")

            theta_ic = np.linspace(theta_min, theta_max, 3)
            psi_ic = np.linspace(psi_min, psi_max, 10)

            initial_conditions = [
                [theta_init, psi_init] for theta_init, psi_init in product(theta_ic, psi_ic)
            ]

        empty_fixed_points = np.empty((len(initial_conditions), len(initial_conditions[0])))
        empty_fixed_points[:] = np.nan

    return bounds, initial_conditions, empty_fixed_points
