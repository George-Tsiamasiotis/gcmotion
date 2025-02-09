import numpy as np
from matplotlib.axes import Axes

from gcmotion.utils.logger_setup import logger

from gcmotion.utils.plot_utils import _pi_mod
from gcmotion.entities.particle import Particle
from gcmotion.plot._base._config import _ParticlePoloidalDrift


def _base_particle_poloidal_drift(particle: Particle, ax: Axes, **kwargs):
    # TODO: Write Documentation

    # Unpack parameters
    config = _ParticlePoloidalDrift()
    for key, value in kwargs.items():
        setattr(config, key, value)

    # This suffix is used to yank the correct attributes from particle.
    suffix = (
        "NU" if config.units == "NU" else "" if config.units == "SI" else ""
    )
    logger.info(
        "Plotting time evolutions in "
        + f"{"NU" if suffix == "NU" else "SI"}..."
    )

    # Make sure percentage is a valid number
    if config.percentage < 1 or config.percentage > 100:
        config.percentage = 100
        logger.warning("Invalid percentage: Plotting the whole thing...")
    points = int(
        np.floor(particle.t_solve.shape[0] * config.percentage / 100) - 1
    )
    theta = particle.theta[:points]
    psi = particle.psi.to(config.flux_units)[:points]

    scatter_kw = {
        "s": config.s,
        "color": config.color,
        "marker": config.marker,
    }

    # Mod theta and plot
    theta = _pi_mod(theta, config.thetalim)[0]
    ax.scatter(theta, psi, **scatter_kw)

    # This pulls its values from _config
    initial_kw = {
        "s": config.init_s,
        "color": config.init_color,
        "marker": config.init_marker,
    }
    if config.initial:
        ax.scatter(theta[0], psi[0], **initial_kw)
