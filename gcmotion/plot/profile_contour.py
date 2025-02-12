import matplotlib.pyplot as plt

from gcmotion.entities.profile import Profile
from gcmotion.plot._base._base_profile_energy_contour import (
    _base_profile_energy_contour,
)
from gcmotion.plot._base._base_profile_pzeta_contour import (
    _base_profile_pzeta_contour,
)
from gcmotion.plot._base._base_contour_colorbar import _base_contour_colorbar
from gcmotion.configuration.plot_parameters import (
    ProfileEnergyContourConfig,
    ProfilePzetaContourConfig,
)

from gcmotion.utils.logger_setup import logger


def profile_energy_contour(profile: Profile, **kwargs):
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
    ycoord: {"psi", "Ptheta"}, optional
        Which coordinate to use for the y axis. Defaults to "psi".
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

    # Make fig more square
    if config.projection == "polar":
        logger.debug("\tPolar projection.")
        config.figsize = (1.2 * config.figsize[1], config.figsize[1])

    # Create figure
    fig_kw = {
        "figsize": config.figsize,
        "dpi": config.dpi,
        "layout": config.layout,
        "facecolor": config.facecolor,
    }
    fig = plt.figure(**fig_kw)
    contourax = fig.add_subplot(projection=config.projection)

    # Draw the contour and get the contour object
    Contour = _base_profile_energy_contour(
        profile=profile, ax=contourax, **kwargs
    )

    # ========
    # Colorbar
    # ========
    # 'cax=None' creates a new axes for the colorbar, by stealing space from
    # the `ax=contour`
    cbar = fig.colorbar(Contour, cax=None, ax=contourax)

    # Now that the colorbar is created, pass its "ax" to be customized
    _base_contour_colorbar(ax=cbar.ax, contour=Contour, numticks=10)

    # Add the title on the cbar's ax
    cbar.ax.set_title(
        label=f"Energy [{config.E_units}]", size=config.cbarlabelsize
    )
    if config.show:
        logger.info("--> Energy contour successfully plotted.")
        plt.show()
    else:
        logger.info("--> Energy contour returned without plotting")
        plt.clf()


def profile_pzeta_contour(profile, **kwargs):
    r"""Plots the Profile's :math:`P_\zeta` contour plot.

    Parameters
    ----------
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

    """

    # Unpack parameters
    config = ProfilePzetaContourConfig()
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
    contourax = fig.subplots()

    # Make fig more square
    if config.projection == "polar":
        config.figsize = (1.2 * config.figsize[1], config.figsize[1])

    # Draw the contour and get the contour object
    logger.debug("\tCalling base contour...")
    Contour = _base_profile_pzeta_contour(
        profile=profile, ax=contourax, **kwargs
    )

    # Colorbar
    # 'cax=None' creates a new axes for the colorbar, by stealing space from
    # the `ax=contour`
    cbar = fig.colorbar(Contour, cax=None, ax=contourax)

    # Now that the colorbar is created, pass its "ax" to be customized
    logger.debug("\tCalling base cbar...")
    _base_contour_colorbar(
        ax=cbar.ax, contour=Contour, numticks=config.numticks
    )

    # Add the title on the cbar's ax
    cbar.ax.set_title(
        label=f"Pzeta [{config.flux_units}]", size=config.cbarlabelsize
    )

    if config.show:
        logger.info("--> Pzeta Contour successfully plotted.")
        plt.show()
    else:
        logger.info("--> Pzeta Contour returned without plotting")
        plt.clf()
