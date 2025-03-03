r"""
Electric Field Profile
----------------

Plots the E(ψ) and Φ(ψ) plots.
"""

import numpy as np
import matplotlib.pyplot as plt

from gcmotion.utils.logger_setup import logger
from gcmotion.entities.tokamak import Tokamak
from gcmotion.entities.profile import Profile
from gcmotion.entities.particle import Particle

from gcmotion.configuration.plot_parameters import EfieldProfileConfig


def efield_profile(entity: Tokamak | Profile | Particle, **kwargs):
    r"""Plots :math:`E(\psi)` and :math:`\Phi(\psi)`.

    Parameters
    ----------
    entity : Tokamak, Profile, Particle
       The object to plot the electric field and potential of.

    Other Parameters
    ----------------
    span : list, optional
         The x-axis zoom limits, relative to :math:`\psi_{wall}`. Defaults to
         [0, 1.1].

    """
    logger.info("==> Plotting electric field profile...")

    # Unpack parameters
    config = EfieldProfileConfig()
    for key, value in kwargs.items():
        setattr(config, key, value)

    # Grab efield object and needed attributes
    efield = entity.efield
    psi_wallNU = entity.psi_wallNU

    # Setup figure
    fig_kw = {
        "figsize": config.figsize,
        "dpi": config.dpi,
        "layout": config.layout,
        "facecolor": config.facecolor,
    }
    fig = plt.figure(**fig_kw)
    ax_dict = fig.subplot_mosaic([["q", "p"]])
    axf, axp = ax_dict["q"], ax_dict["p"]
    axf.margins(x=0, y=0.02)
    axp.margins(x=0, y=0.01)

    # Calculate values
    psi = psi_wallNU * np.linspace(config.span[0], config.span[1], config.points)
    logger.trace(psi)
    _el_field = efield.Er(psi.m)
    _el_potential = efield.PhiNU(psi.m, 0)
    logger.trace(_el_field.shape)
    logger.trace(_el_potential.shape)
    logger.trace(psi_wallNU)

    # Convert to requaested units
    Q = entity.Q
    el_field = Q(_el_field, "NUVolts / NUmeter").to(config.field_units).m
    el_potential = Q(_el_potential, "NUVolts").to(config.potential_units).m

    # Plot
    axf.plot(psi / psi_wallNU, el_field)
    axp.plot(psi / psi_wallNU, el_potential)

    # Add vertical lines indicating the wall
    if config.span[1] >= 1:
        axf.axvline(x=1, color=config.wall_color)
        axp.axvline(x=1, color=config.wall_color)

    # Labels
    axf.set_title(r"$E(\psi)$", size=config.ax_title_size)
    axf.set_xlabel(r"$\psi/\psi_{wall}$", size=config.labelsize)
    axf.set_ylabel(rf"$E(\psi) [{config.field_units}]$", size=config.labelsize)

    axp.set_title(r"$\Phi(\psi)$", size=config.ax_title_size)
    axp.set_xlabel(r"$\psi/\psi_{wall}$", size=config.labelsize)
    axp.set_ylabel(rf"$\Phi(\psi) [{config.potential_units}]$", size=config.labelsize)

    # Format the cursor so as to show the actual psi value
    def fmt_E(x, y):
        return f"ψ={x*psi_wallNU.m:.4g}[NU], E={y:.4g}[{config.field_units}]"

    def fmt_Phi(x, y):
        return f"ψ={x*psi_wallNU.m:.4g}[NU], Φ={y:.4g}[{config.potential_units}]"

    axf.format_coord = fmt_E
    axp.format_coord = fmt_Phi

    if config.show:
        logger.info("--> Electric Field profile successfully plotted.")
        plt.show()
    else:
        logger.info("--> Electric Field profile returned without plotting")
        plt.clf()
