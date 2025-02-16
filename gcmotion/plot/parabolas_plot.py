"""
Simple script that draws parabolas diagram along with the trapped passing boundary
if asked.
"""

import numpy as np
from collections import deque

from gcmotion.utils.logger_setup import logger
import matplotlib.pyplot as plt

from gcmotion.configuration.plot_parameters import ParabolasPlotConfig
from gcmotion.entities.profile import Profile
from gcmotion.scripts.parabolas import calc_parabolas
from gcmotion.plot._base._base_parabolas_tpb_plot import _plot_parabolas_tpb


def parabolas_diagram(profile: Profile, **kwargs):
    r"""

    This script draw the parabolas diagram along with the trapped passing boundary
    (if asked) by plotting the values calculated in :py:func:`_calc_parabolas`.

    Parameters
    ----------
    profile : Profile
        Profile object that contains Tokamak information like bfield, mu,
        useful psi values.

    Other Parameters
    ----------
    Pzetalim : tuple,list, optional
        The Pzeta limits within which the RW, LW, MA parabolas' values are to
        be calculated. CAUTION: the limits must be normalized to psip_wallNU.
        Defaults to (-1.5,1).
    Pzeta_density : int, optional
        The density of Pzeta points that will be used to calculate
        the RW, LW, MA parabolas' values. Defaults to 1000.
    enlim : tuple, list, optional
        The Pzeta limits within which the RW, LW, MA parabolas' values are to
        be calculated. CAUTION: the limits must be normalized to E/:math:`\mu B_0`.
        Defaults to (0,3).
    TPB : bool, optional
        Boolean that determines weather the trapped-passing boundary is to
        be plotted. Defaults to ``False``.

    Notes
    -----
    For a full list of all available optional parameters, see the dataclass
    ParabolasPlotConfig at gcmotion/config/plot_parameters. The defaults values
    are set there, and are overwritten if passed as arguements.
    """

    # Unpack parameters
    config = ParabolasPlotConfig()
    for key, value in kwargs.items():
        setattr(config, key, value)

    # Create figure
    fig_kw = {
        "figsize": config.figsize,
        "dpi": config.dpi,
        "layout": config.layout,
        "facecolor": config.facecolor,
    }

    fig, par_ax = plt.subplots(1, 1, **fig_kw)

    logger.info("Creating parabolas diagram figure.")

    # Calculate parabolas values
    x, x_TPB, y_R, y_L, y_MA, TPB_O, TPB_X, v_R, v_L, v_MA = calc_parabolas(
        Pzetalim=config.Pzetalim,
        profile=profile,
        Pzeta_density=config.Pzeta_density,
        calc_TPB=config.plot_TPB,
    )

    logger.info(
        f"Successfully calculated {y_L.shape[0]} parabolas values with vertices RW: {v_R}, LW: {v_L}, MA: {v_MA}"
    )

    # Plot right left boundar, magnetic axis
    par_ax.plot(
        x,
        y_R,
        linestyle="dashed",
        color=config.parabolas_color,
        label="RW",
        linewidth=config.linewidth,
    )
    par_ax.plot(
        x,
        y_L,
        linestyle="dashdot",
        color=config.parabolas_color,
        label="LW",
        linewidth=config.linewidth,
    )
    par_ax.plot(
        x,
        y_MA,
        linestyle="dotted",
        color=config.parabolas_color,
        label="MA",
        linewidth=config.linewidth,
    )

    # Plot vertical dashed line to set tpb left limit if asked
    if config.show_d_line:
        par_ax.plot(
            [v_R[0], v_L[0]],
            [v_R[1], v_L[1]],
            color=config.d_line_color,
            linestyle="dashed",
            linewidth=config.d_linewidth,
            alpha=config.d_line_alplha,
        )

    logger.info("Plotted RW, LW, MA parabolas.")

    # If asked plot trapped-passing boundary
    if config.plot_TPB:
        _plot_parabolas_tpb(
            profile=profile, X_energies=TPB_X, O_energies=TPB_O, ax=par_ax, x_TPB=x_TPB, **kwargs
        )

        logger.info(
            f"Plotted {profile.bfield.plain_name} trapped-passing boundary in parabolas diagram."
        )

    par_ax.set_xlabel(
        r"$\dfrac{P_\zeta}{\psi_{p_w}}$",
        rotation=config.xlabel_rotation,
        fontsize=config.xlabel_fontsize,
    )
    par_ax.set_ylabel(
        r"$\dfrac{E}{\mu B_0}$", rotation=config.ylabel_rotation, fontsize=config.ylabel_fontsize
    )

    par_ax.set_title(
        f"Parabolas Diagram ({profile.bfield.plain_name})",
        fontsize=config.title_fontsize,
        color=config.title_color,
    )
    par_ax.set_ylim(config.enlim)

    if config.parabolas_legend:
        par_ax.legend()

    plt.show()
    plt.ion()
