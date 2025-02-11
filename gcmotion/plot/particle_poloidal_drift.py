r"""
Plots the particle's poloidal drift on top of the energy contour plot.
"""

import matplotlib.pyplot as plt
from dataclasses import asdict

from gcmotion.entities.particle import Particle
from gcmotion.configuration.plot_parameters import ParticlePoloidalDrift
from gcmotion.plot._base._base_profile_Energy_contour import (
    _base_profile_Energy_contour,
)
from gcmotion.plot._base._base_contour_colorbar import _base_contour_colorbar
from gcmotion.plot._base._base_particle_poloidal_drift import (
    _base_particle_poloidal_drift,
)
from gcmotion.utils.plot_utils import _auto_yspan

from gcmotion.utils.logger_setup import logger


def particle_poloidal_drift(particle: Particle, **kwargs):
    r"""Plots the particle's poloidal drift on top of the energy contour
    plot.

    Parameters
    ----------
    particle: Particle
        The current working particle

    Other Parameters
    ----------------
    thetalim: list, optional
        The :math:`\theta` span. Ignored when plotting in polar projection.
        Defaults to [-π, π].
    psilim: list, optional
        The :math:`\psi` span, relative to :math:`\psi_{wall}`. Defaults to [0,
        1.2]
    projection: {None, "polar"}, optional
        The plot projection. None means cartesian. Defaults to None.
    levels: int, optional
        The energy contour levels. Defaults to 30.
    E_units: str, optional
        The Energy units, as read by pint. Defaults to "keV".
    flux_units: str, optional
        The magnetic flux units, as read by pint. Defaults to "Tesla *
        meter^2".
    grid_density: int, optional
        The contour's grid density. Defaults to 200.
    potential: bool, optional
        Whether or not to add the electric potential in the calculation of
        energy. Defaults to True.
    wall: bool, optional
        Whether or not to shade the area :math:`\psi>\psi_{wall}`. Defaults to
        True.
    cursor: bool, optional
        Whether or not to add cursor data to the plot. This is a mild
        performance hit. Defaults to True.

    """
    logger.info("==> Plotting Particle Poloidal Drift...")

    # Unpack parameters
    config = ParticlePoloidalDrift()
    for key, value in kwargs.items():
        setattr(config, key, value)

    # Make fig more square
    if config.projection == "polar":
        config.figsize = (1.2 * config.figsize[1], config.figsize[1])

    # Create figure
    fig_kw = {
        "figsize": config.figsize,
        "dpi": config.dpi,
        "layout": config.layout,
        "facecolor": config.facecolor,
    }
    fig = plt.figure(**fig_kw)
    driftax = fig.add_subplot(projection=config.projection)

    psi = particle.psi.to(config.flux_units)

    # Set up ylim now to pass to contour as well
    psi_wall = particle.profile.psi_wall.to("psi_wall")
    if config.psilim == "auto":
        logger.trace(f"psilim passed: {config.psilim}")
        config.psilim = particle.Q(
            _auto_yspan(psi.to("psi_wall").m, psi_wall.m)
        ).m
        logger.trace(f"Auto psilim: {config.psilim}")

    # ==============
    # Energy Contour
    # ==============
    # Draw the contour and get the contour object
    logger.debug("\tCalling base contour...")
    Contour = _base_profile_Energy_contour(
        profile=particle.profile, ax=driftax, **asdict(config)
    )

    # ==============
    # Poloidal Drift
    # ==============
    logger.debug("\tCalling base poloidal drift...")
    _base_particle_poloidal_drift(particle=particle, ax=driftax, **kwargs)

    # ========
    # Colorbar
    # ========
    # 'cax=None' creates a new axes for the colorbar, by stealing space from
    # the `ax=contour`
    cbar = fig.colorbar(Contour, cax=None, ax=driftax)

    # Now that the colorbar is created, pass its "ax" to be customized
    logger.debug("\tCalling base cbar...")
    E = particle.ENU.to(config.E_units)
    _base_contour_colorbar(
        ax=cbar.ax, contour=Contour, numticks=config.numticks, energy_line=E
    )

    cbar.ax.set_title(
        label=f"Energy [{config.E_units}]", size=config.cbarlabelsize
    )

    if config.show:
        logger.info("--> Particle Poloidal Drift successfully plotted.")
        plt.show()
    else:
        logger.info("--> Particle Poloidal Drift returned without plotting.")
        plt.clf()
