import matplotlib.pyplot as plt
from gcmotion.plotters.fixed_points_plot import fixed_points_plot

from gcmotion.entities.profile import Profile
from gcmotion.plot._base._base_profile_Energy_contour import (
    _base_profile_Energy_contour,
)
from gcmotion.plot._base._base_profile_Pzeta_contour import (
    _base_profile_Pzeta_contour,
)
from gcmotion.plot._base._base_contour_colorbar import _base_contour_colorbar
from gcmotion.configuration.plot_parameters import (
    ProfileEnergyContourConfig,
    ProfilePzetaContourConfig,
)

from gcmotion.utils.logger_setup import logger


def fixed_points_Energy_contour(profile: Profile, **kwargs):
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

    """
    logger.info("==> Plotting profile's energy contour...")

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
    fixed_points_plot(
        profile=profile,
        ax=contourax,
        **kwargs,
    )

    # Draw the contour and get the contour object
    Contour = _base_profile_Energy_contour(profile=profile, ax=contourax, **kwargs)

    # ========
    # Colorbar
    # ========
    # 'cax=None' creates a new axes for the colorbar, by stealing space from
    # the `ax=contour`
    cbar = fig.colorbar(Contour, cax=None, ax=contourax)

    # Now that the colorbar is created, pass its "ax" to be customized
    _base_contour_colorbar(ax=cbar.ax, contour=Contour, numticks=10)

    # Add the title on the cbar's ax
    cbar.ax.set_title(label=f"Energy [{config.E_units}]", size=config.cbarlabelsize)
    if config.show:
        logger.info("--> Energy contour successfully plotted.")
        plt.show()
    else:
        logger.info("--> Energy contour returned without plotting")
        plt.clf()
