import matplotlib.pyplot as plt
from gcmotion.plot._base._base_fixed_points_profile_contour import _base_fixed_points_plot

from gcmotion.entities.profile import Profile
from gcmotion.plot._base._base_profile_energy_contour import (
    _base_profile_energy_contour,
)
from gcmotion.plot._base._base_contour_colorbar import _base_contour_colorbar
from gcmotion.configuration.plot_parameters import (
    ProfileEnergyContourConfig,
)

from gcmotion.utils.logger_setup import logger


def fixed_points_energy_contour(profile: Profile, **kwargs):
    r"""Plots the Profile's energy contour plot.

    Parameters
    ----------
    profile: Profile
        The Profile entity.

    Other Parameters
    ----------------
    thetalim : list, optional
        The :math:`\theta` span in radians. Defaults to [-π,π].
    psilim : list, optional
        The :math:`\psi` span in units of *psi_wall*. Defaults to [0, 1.2].
    levels : int, optional
        The number of contour lines. Defaults to 25.
    flux_units : str, optional
        The units of the psi/Ptheta axis. Can be "Magnetic_flux"(same as "Tesla
        * meters^2"), "NUmagnetic_flux" (same as "NUTesla * NUmeters^2"),
        "psi_wall", "NUpsi_wall", or any other pint unit with the same
        dimensionality. Defaults to "Tesla * meters^2".
    E_units : str, optional
        The Energy units. Can be "eV", "keV", "Joule", "NUJoule" or any other
        pint unit with the same dimensionality. Defaults to "keV".
    potential : bool, optional
        Whether or not to add the potential Φ term when calculating the Energy.
        Defaults to True.
    wall : bool, optional
        Whether or not to shade the area above :math:`\psi_wall`. Defaults to
        True.
    grid_density: int, optional
        The contour's grid density. Defaults to 200.

    Fixed Points Parameters
    ----------------
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
        Boolean determining weather fixed points' information is to be is to be printed in the log. Defaults to ``False``.
    ic_info : bool, optional
        Boolean determing weather information on the initial condition is to be is to be printed in the log.
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
    """
    logger.info("==> Plotting profile's energy contour with fixed points...")

    # Unpack parameters
    config = ProfileEnergyContourConfig()
    for key, value in kwargs.items():
        setattr(config, key, value)

    # Create figure
    fig_kw = {
        "figsize": config.figsize,
        "dpi": config.dpi,
        "layout": config.layout,
        "facecolor": config.facecolor,
    }
    fig = plt.figure(**fig_kw)
    contourax = fig.add_subplot(projection=config.projection)

    # Here will go fixed points plot
    _base_fixed_points_plot(
        profile=profile,
        ax=contourax,
        **kwargs,
    )

    # Draw the contour and get the contour object
    Contour = _base_profile_energy_contour(profile=profile, ax=contourax, **kwargs)

    # ========
    # Colorbar
    # ========
    # 'cax=None' creates a new axes for the colorbar, by stealing space from
    # the `ax=contour`
    cbar = fig.colorbar(Contour, cax=None, ax=contourax)

    # Now that the colorbar is created, pass its "ax" to be customized
    _base_contour_colorbar(ax=cbar.ax, contour=Contour, numticks=10)

    contourax.set_ylim(profile.Q(config.psilim, "NUpsi_wall").to(config.flux_units))

    # Add the title on the cbar's ax
    cbar.ax.set_title(label=f"Energy [{config.E_units}]", size=config.cbarlabelsize)
    if config.show:
        logger.info("--> Energy contour with fixed points successfully plotted.")
        plt.show()
    else:
        logger.info("--> Energy contour with fixed points returned without plotting")
        plt.clf()
