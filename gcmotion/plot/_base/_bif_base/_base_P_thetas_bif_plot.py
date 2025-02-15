import numpy as np
from collections import deque
from gcmotion.utils.logger_setup import logger

from gcmotion.utils.bif_values_setup import set_up_bif_plot_values
from gcmotion.plot._base._bif_base._bif_config import _PThetasFixedPlotConfig


def _P_thetas_bif_plot(
    profiles: list,
    X_P_thetas: list | deque,
    O_P_thetas: list | deque,
    which_COM: str,
    ax=None,
    **kwargs,
):
    r"""Base plotting function. Only draws upon a given axis without showing
    any figures.

    Parameters
    ----------
    profiles : list, deque
        List of profile objects.
    X_P_thetas : deque, list
        The values of the :math:`P_{\theta}`s of the X points for each COM value.
    O_P_thetas : deque, list
        The values of the :math:`P_{\theta}`s of the O points for each COM value.
    which_COM : str
        String that indicates with respect to which constant of motion (COM) :math:`\mu`
        or :math:`P_{\zeta}` the :math:`P_{\theta}`s fixed are plotted.
    ax : Axes
        The ax upon which to draw.

    Notes
    -----
    For a full list of all available optional parameters, see the dataclass
    _PPThetasFixedPlotConfig at gcmotion/plot/_base/_bif_base/_bif_config.
    The defaults values are set there, and are overwritten if passed as arguements.

    """
    logger.info("\t==> Plotting Base P_thetas Fixed Bifurcation Diagram...")

    # Unpack parameters
    config = _PThetasFixedPlotConfig()
    for key, value in kwargs.items():
        setattr(config, key, value)

    # P_theta Fixed Bifurcation
    COM_plotX, X_P_theta_plot = set_up_bif_plot_values(
        profiles=profiles, y_values=X_P_thetas, which_COM=which_COM
    )
    COM_plotO, O_P_theta_plot = set_up_bif_plot_values(
        profiles=profiles, y_values=O_P_thetas, which_COM=which_COM
    )

    # Set the upper limit of the y axis properly
    # Combine the two lists for comparison
    psi_wallNU = profiles[0].psi_wall.to("NUMagnetic_flux")
    P_theta_wallNU = profiles[0].findPtheta(psi=psi_wallNU, units="NUCanonical_momentum").m

    combined_P_theta_plot = X_P_theta_plot + O_P_theta_plot
    ul = max(combined_P_theta_plot) * 1.05

    if np.isclose(max(combined_P_theta_plot), P_theta_wallNU):
        ul = P_theta_wallNU * 1.05

    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    ax.set_ylabel(
        r"$P_{\theta_s}$ Fixed",
        rotation=config.Ptheta_ylabel_rotation,
        fontsize=config.Ptheta_ylabel_fontsize,
    )

    ax.set_ylim(0, ul)

    ax.scatter(
        COM_plotX,
        X_P_theta_plot,
        marker=config.Pthetas_X_marker,
        s=config.Pthetas_X_markersize,
        color=config.Pthetas_X_markercolor,
        label="X points",
    )

    ax.scatter(
        COM_plotO,
        O_P_theta_plot,
        marker=config.Pthetas_O_marker,
        s=config.Pthetas_O_markersize,
        color=config.Pthetas_O_markercolor,
        label="O points",
    )

    ax.axhline(
        y=P_theta_wallNU,
        color=config.wall_line_color,
        linestyle=config.wall_linestyle,
        linewidth=config.wall_linewidth,
        alpha=config.wall_line_alpha,
    )

    if config.Ptheta_legend:
        ax.legend(loc=config.Ptheta_legend_loc)
