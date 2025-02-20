r"""
Draws fixed points plot. This method is called internally by ``profile_contour()``
as well.
"""

from gcmotion.scripts.fixed_points import fixed_points as get_fixed_points
from gcmotion.entities.profile import Profile

from gcmotion.utils.XO_points_classification import XO_points_classification as xoc
from gcmotion.plot._base._config import _FixedPointsPlotConfig

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

from time import time

from gcmotion.utils.logger_setup import logger


def _base_fixed_points_plot(
    profile: Profile,
    ax: Axes = None,
    **kwargs,
):
    r"""

        :meta public:

    Function that creates a plot of the GC Hamiltonian's fixed points. Most of its arguments
    will be passed into :py:func:`fixed_points`. This function is almost always used in
    :py:func:`profile_contour`, in order to visualize the results.

            Parameters
            ----------
            profile : Profile
                Profile object that contains Tokamak and Particle information.
            ax : Axes, optional
                The axes upon which the plot is to be realized.
            Other Parameters
            ----------
            psilim : list, optional
                Limits of the of the :math:`\theta`, :math:`\psi` search area with respect
                to the :math:`\psi` variable. Defaults to [0.01 , 1.8]. CUTION: The limits are given
                normalized to :math:`\psi_{wall}`.
            thetalim : list, optional
                Limits of the of the :math:`\theta`, :math:`\psi` search area with respect
                to the :math:`\theta` variable. Defaults to [-:math:`\pi`, :math:`\pi`].
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
                Boolean determining weather fixed points' information is to be printed. Defaults to ``False``.
            ic_info : bool, optional
                Boolean determing weather information on the initial condition is to be printed.
                Defaults to ``False``.
            plot_init_cond : bool, optional
                Boolean that determines weather the initial conditions passed into :py:func:`fixed_points`
                will be plotted. Defaults to ``False``.
            LAR_thetas : bool, optional
                Boolean determining weather the theta values for which fixed points occur are to be
                considered known (LAR thetas are 0 and :math:`\pi`). Defaults to ``False``.
            only_confined : bool, optional
                Boolean determining if the search for :math:`\psi_{fixed}` will be conducted only for
                :math:`\psi` < :math:`\psi_{wall}` (confined particles). Defaults to ``False``.

            Notes
            -----
            For a full list of all available optional parameters, see the dataclass
            _FixedPointsPlotConfig at gcmotion/plot/_base/_config. The default values
            are set there, and are overwritten if passed as arguements.
    """

    # Unpack Parameters
    config = _FixedPointsPlotConfig()
    for key, value in kwargs.items():
        setattr(config, key, value)

    # Check axes
    if ax is None:
        fig_kw = {
            "figsize": config.figsize,
            "dpi": config.dpi,
            "layout": config.layout,
            "facecolor": config.facecolor,
        }
        fig, ax = plt.subplots(1, 1, **fig_kw)

        logger.info("Axes was not given for fixed point plot. Creating figure")

    # Determine output units
    output_units = config.flux_units

    start = time()
    # Calculate fixed points
    _, fixed_points, initial_conditions = get_fixed_points(profile=profile, **kwargs)
    logger.info(
        f"Calculated fixed points ['NUMagnetic_flux'] for fixed_points_plot with Pz={profile.PzetaNU}, mu={profile.muNU} in {(time() - start):.1f}s"
    )
    print(f"\n FIXED POINTS RUN IN {(time() - start):.1f}s\n")

    # CAUTION: The xoc function takes in psis_fixed but returns P_thetas_fixed if ssked
    X_points, O_points = xoc(
        unclassified_fixed_points=fixed_points, profile=profile, to_P_thetas=False
    )

    # Convert deque to numpy arrays for easy manipulation
    X_thetas, X_psisNU = zip(*X_points) if X_points else ([], [])
    O_thetas, O_psisNU = zip(*O_points) if O_points else ([], [])

    logger.info(f"Classified fixed points and XPoints={X_points}, OPoints={O_points} ")

    X_psis = profile.Q(X_psisNU, "NUMagnetic_flux").to(output_units)
    O_psis = profile.Q(O_psisNU, "NUMagnetic_flux").to(output_units)

    logger.info(f"Converted fixed points from 'NUMagnetic_flux' to {output_units} ")

    ax.set_xticks([-np.pi, 0, np.pi])
    ax.set_xticklabels([r"$-\pi$", "0", r"$\pi$"])
    ax.set_xlim([-np.pi, np.pi])

    ax.scatter(X_thetas, X_psis.m, marker="x", color="#80FF80", s=100)
    ax.scatter(O_thetas, O_psis.m, marker="o", edgecolor="yellow", facecolors="none", s=100)

    logger.info(
        f"Plotted fixed points for fixed_points_plot with Pz={profile.PzetaNU} and mu={profile.muNU}"
    )

    if config.fp_plot_init_cond:

        thetas_init, psis_initNU = zip(*initial_conditions) if initial_conditions else ([], [])

        psis_init = profile.Q(psis_initNU, "NUmagnetic_flux").to(output_units)
        ax.scatter(
            thetas_init,
            psis_init.m,
            marker=config.ic_marker,
            color=config.ic_markercolor,
            s=config.ic_markersize,
            label="Initial Guesses",
        )

        ax.legend()

        logger.info(f"Plotted initial conditions for fixed points")
