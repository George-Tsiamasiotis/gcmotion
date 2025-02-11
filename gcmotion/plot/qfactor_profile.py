r"""
Qfactor Profile
----------------

Plots the q(ψ) and ψ_p(ψ) plots.
"""

import numpy as np
import matplotlib.pyplot as plt

from gcmotion.utils.logger_setup import logger
from gcmotion.entities.tokamak import Tokamak
from gcmotion.entities.profile import Profile
from gcmotion.entities.particle import Particle

from gcmotion.configuration.plot_parameters import QfactorProfileConfig


def qfactor_profile(entity: Tokamak | Profile | Particle, **kwargs):
    r"""Plots :math:`q(\psi)` and :math:`\psi_p(\psi)`.

    Parameters
    ----------
    entity : Tokamak, Profile, Particle
       The object to plot the qfactor of.

    Other Parameters
    ----------------
    span : list, optional
         The x-axis zoom limits, relative to :math:`\psi_{wall}`. Defaults to
         [0, 1.1].

    """
    logger.info("==> Plotting qfactor profile...")

    # Unpack parameters
    config = QfactorProfileConfig()
    for key, value in kwargs.items():
        setattr(config, key, value)

    # Grab qfactor object and needed attributes
    qfactor = entity.qfactor
    psi_wallNU = entity.psi_wallNU
    if hasattr(qfactor, "is_numerical") and config.span[1] > 1:
        logger.warning(
            "\tNumerical qfactor: span[1] > 1 is an extrapolation of the data"
        )

    # Setup figure
    fig_kw = {
        "figsize": config.figsize,
        "dpi": config.dpi,
        "layout": config.layout,
        "facecolor": config.facecolor,
    }
    fig = plt.figure(**fig_kw)
    ax_dict = fig.subplot_mosaic([["q", "p"]])
    axq, axp = ax_dict["q"], ax_dict["p"]
    axq.margins(0)
    axp.margins(0)

    # Calculate values
    psi = psi_wallNU * np.linspace(
        config.span[0], config.span[1], config.points
    )
    logger.trace(psi)
    q = qfactor.solverqNU(psi.m)
    psip = qfactor.psipNU(psi.m)
    logger.trace(q.shape)
    logger.trace(psi_wallNU)

    # Plot
    axq.plot(psi / psi_wallNU, q)
    axp.plot(psi / psi_wallNU, psip)

    # Add vertical lines indicating the wall
    if config.span[1] >= 1:
        axq.axvline(x=1, color=config.wall_color)
        axp.axvline(x=1, color=config.wall_color)
        axq.axhline(
            y=qfactor.solverqNU(psi_wallNU.m),
            color=config.qwall_color,
            linestyle=config.qwall_style,
        )

    # Labels
    axq.set_title(r"$q(\psi)$", size=config.ax_title_size)
    axq.set_xlabel(r"$\psi/\psi_{wall}$", size=config.labelsize)
    axq.set_ylabel(r"$q(\psi)$", size=config.labelsize)

    axp.set_title(r"$\psi_p(\psi)$", size=config.ax_title_size)
    axp.set_xlabel(r"$\psi/\psi_{wall}$", size=config.labelsize)
    axp.set_ylabel(r"$\psi_p(\psi) [NU]$", size=config.labelsize)

    # Format the cursor so as to show the actual psi value
    def fmt_q(x, y):
        return f"ψ={x*psi_wallNU.m:.4g}[NU], q={y:.4g}"

    def fmt_psip(x, y):
        return f"ψ={x*psi_wallNU.m:.4g}[NU], ψ_p={y:.4g}[NU]"

    axq.format_coord = fmt_q
    axp.format_coord = fmt_psip

    if config.show:
        logger.info("--> QFactor profile successfully plotted.")
        plt.show()
    else:
        logger.info("--> QFactor profile returned without plotting")
        plt.clf()
