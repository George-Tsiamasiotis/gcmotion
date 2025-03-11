import numpy as np
from collections import deque
from gcmotion.utils.logger_setup import logger

from gcmotion.plot._base._bif_base._bif_config import _ThetasFixedPlotConfig
from gcmotion.scripts.fixed_points_bif.bif_values_setup import set_up_bif_plot_values


def _thetas_bif_plot(
    profiles: list,
    X_thetas: list | deque,
    O_thetas: list | deque,
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
    X_thetas : deque, list
        The values of the :math:`\theta`s of the X points for each COM value.
    O_thetas : deque, list
        The values of the :math:`\theta`s of the O points for each COM value.
    which_COM : str
        String that indicates with respect to which constant of motion (COM) :math:`\mu`
        or :math:`P_{\zeta}` the :math:`\theta`s fixed are plotted.
    ax : Axes
        The ax upon which to draw.

    Notes
    -----
    For a full list of all available optional parameters, see the dataclass
    _ThetasFixedPlotConfig at gcmotion/plot/_base/_bif_base/_bif_config.
    The defaults values are set there, and are overwritten if passed as arguements.

    """
    logger.info("\t==> Plotting Base Thetas Fixed Bifurcation Diagram...")

    # Unpack parameters
    config = _ThetasFixedPlotConfig()
    for key, value in kwargs.items():
        setattr(config, key, value)

    # Theta Fixed Bifurcation
    COM_plotX, X_theta_plot = set_up_bif_plot_values(
        profiles=profiles, y_values=X_thetas, which_COM=which_COM
    )
    COM_plotO, O_theta_plot = set_up_bif_plot_values(
        profiles=profiles, y_values=O_thetas, which_COM=which_COM
    )

    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    ax.set_ylabel(
        r"$\theta_s$ Fixed",
        rotation=config.theta_ylabel_rotation,
        fontsize=config.theta_ylabel_fontsize,
    )

    ax.set_yticks([-np.pi, 0, np.pi])
    ax.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])
    ax.set_ylim([-np.pi - 0.5, np.pi + 0.5])

    ax.scatter(
        COM_plotX,
        X_theta_plot,
        marker=config.thetas_X_marker,
        s=config.thetas_X_markersize,
        color=config.thetas_X_markercolor,
        label="X Points",
    )
    ax.scatter(
        COM_plotO,
        O_theta_plot,
        marker=config.thetas_O_marker,
        s=config.thetas_O_markersize,
        color=config.thetas_O_markercolor,
        label="O Points",
    )

    if config.theta_legend:
        ax.legend()
