r"""
Ptheta - Psi plot
-----------------

Plots Pθ(ψ).

"""

import numpy as np
import matplotlib.pyplot as plt

from gcmotion.utils.logger_setup import logger
from gcmotion.entities.profile import Profile
from gcmotion.entities.particle import Particle

from gcmotion.configuration.plot_parameters import PsiPthetaConfig


def psi_ptheta_plot(entity: Profile | Particle, **kwargs):
    r"""Plots :math:`P_\theta(\psi)`.

    Parameters
    ----------
    entity : Profile, Particle
       The object to plot the :math:`P_\theta(\psi)` of.

    Other Parameters
    ----------------
    flux_units: str, optional
        The :math:`\psi` units. Defaults to "NUMagnetic_flux".
    Ptheta_units: str, optional
        The :math:`P_\theta` units. Defaults to "NUCanonical_momentum".

    Notes
    -----
    While both :math:`\psi's` and :math:`P_\theta's` units are configurable,
    the :math:`P_\theta - \psi` comparison only makes sense in Normalized Units
    (default).

    """
    logger.info("==> Plotting psi-Ptheta plot...")

    # Unpack parameters
    config = PsiPthetaConfig()
    for key, value in kwargs.items():
        setattr(config, key, value)

    if isinstance(entity, Profile):
        findPtheta = entity.findPtheta
        PzetaNU = entity.PzetaNU
    elif isinstance(entity, Particle):
        findPtheta = entity.profile.findPtheta
        PzetaNU = entity.Pzeta0NU

    # Grab qfactor object and needed attributes
    psi_wallNU = entity.psi_wallNU

    # Setup figure
    fig_kw = {
        "figsize": config.figsize,
        "dpi": config.dpi,
        "layout": config.layout,
        "facecolor": config.facecolor,
    }
    fig = plt.figure(**fig_kw)
    ax = fig.add_subplot()
    ax.grid(True)
    ax.margins(0)

    ax.tick_params(axis="x", labelsize=config.x_ticksize)
    ax.tick_params(axis="y", labelsize=config.y_ticksize)

    # Calculate values
    # Configurable span is really not necessary
    _psi = psi_wallNU * np.linspace(0, 1.05, config.points)
    psi = entity.Q(_psi, "NUMagnetic_flux").to(config.flux_units)
    Ptheta = findPtheta(psi, units=config.Ptheta_units)

    # Plot
    ax.plot(
        psi,
        Ptheta,
        lw=config.linewidth,
        label=r"$P_\theta(\psi)$",
    )

    # Add vertical lines indicating the wall
    ax.axvline(x=psi_wallNU, color=config.wall_color)
    ax.axhline(
        y=findPtheta(psi_wallNU, units=config.Ptheta_units).m,
        color=config.wall_color,
        linestyle=config.wall_style,
    )

    # Add y=x line
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(
        lims,
        lims,
        linestyle="dashdot",
        color="k",
        alpha=0.75,
        zorder=-1,
        label="y=x",
    )

    # Labels
    ax.set_title(
        "\t" r"$P_\theta(\psi)$" "\t" r"[$P_\zeta = $" f"{PzetaNU.m:.4g}]",
        size=config.ax_title_size,
    )
    ax.set_xlabel(
        r"$\psi$",  # + f"[{config.flux_units}]",
        size=config.labelsize,
    )
    ax.set_ylabel(
        r"$P_\theta(\psi)$",  # + f"[{config.Ptheta_units}]",
        size=config.labelsize,
    )
    ax.legend()

    # Format the cursor so as to show the actual psi value
    def fmt(x, y):
        return f"ψ={x:.7g},\tPθ={y:.7g}"

    ax.format_coord = fmt

    if config.show:
        logger.info("--> ψ-Ρθ plotted successfully.")
        plt.show()
    else:
        logger.info("--> ψ-Ρθ returned without plotting")
        plt.clf()
