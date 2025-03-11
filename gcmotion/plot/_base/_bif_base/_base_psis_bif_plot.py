import numpy as np
from matplotlib import ticker
from collections import deque
from gcmotion.utils.logger_setup import logger
from gcmotion.entities.profile import Profile

from gcmotion.scripts.fixed_points_bif.bif_values_setup import set_up_bif_plot_values
from gcmotion.plot._base._bif_base._bif_config import _PsisFixedPlotConfig


def _psis_bif_plot(
    profile: Profile,
    COM_values: list | deque,
    X_psis: list | deque,
    O_psis: list | deque,
    which_COM: str,
    ax=None,
    **kwargs,
):
    r"""Base plotting function. Only draws upon a given axis without showing
    any figures.

    Parameters
    ----------
    profile : Profile
        Profile object containing Tokamak information.
    COM_values : list, deque
        List of COM values :math:`P_{\zeta}`'s or :math:`\mu`'s in [NU].
    X_psis : deque, list
        The values of the :math:`\psi`s of the X points for each COM value.
    O_psis : deque, list
        The values of the :math:`\psi`s of the O points for each COM value.
    which_COM : str
        String that indicates with respect to which constant of motion (COM) :math:`\mu`
        or :math:`P_{\zeta}` the :math:`\psi`s fixed are plotted.
    ax : Axes
        The ax upon which to draw.

    Notes
    -----
    For a full list of all available optional parameters, see the dataclass
    _PsisFixedPlotConfig at gcmotion/plot/_base/_bif_base/_bif_config.
    The defaults values are set there, and are overwritten if passed as arguements.

    """
    logger.info("\t==> Plotting Base psis Fixed Bifurcation Diagram...")

    # Unpack parameters
    config = _PsisFixedPlotConfig()
    for key, value in kwargs.items():
        setattr(config, key, value)

    # psi Fixed Bifurcation
    COM_plotX, X_psi_plot = set_up_bif_plot_values(
        profile=profile, COM_values=COM_values, y_values=X_psis, which_COM=which_COM
    )
    COM_plotO, O_psi_plot = set_up_bif_plot_values(
        profile=profile, COM_values=COM_values, y_values=O_psis, which_COM=which_COM
    )

    # Set the upper limit of the y axis properly
    # Combine the two lists for comparison
    psi_wall = profile.psi_wall.to(config.flux_units).m

    combined_psi_plot = X_psi_plot + O_psi_plot
    ul = max(combined_psi_plot) * 1.05

    if np.isclose(max(combined_psi_plot), psi_wall):
        ul = psi_wall * 1.05

    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    ax.set_ylabel(
        rf"$\psi_s$ Fixed [{config.flux_units}]",
        rotation=config.psi_ylabel_rotation,
        fontsize=config.psi_ylabel_fontsize,
    )

    ax.set_ylim(0.95 * min(combined_psi_plot), ul)

    ax.scatter(
        COM_plotX,
        X_psi_plot,
        marker=config.psis_X_marker,
        s=config.psis_X_markersize,
        color=config.psis_X_markercolor,
        label="X points",
    )

    ax.scatter(
        COM_plotO,
        O_psi_plot,
        marker=config.psis_O_marker,
        s=config.psis_O_markersize,
        color=config.psis_O_markercolor,
        label="O points",
    )

    ax.axhline(
        y=psi_wall,
        color=config.wall_line_color,
        linestyle=config.wall_linestyle,
        linewidth=config.wall_linewidth,
        alpha=config.wall_line_alpha,
    )

    if config.psi_legend:
        ax.legend(loc=config.psi_legend_loc)

    # Create secondary Ptheta ax
    logger.debug("\t\tAdding secondary Ptheta ax to bifurcation diagram.")
    ax2 = ax.twinx()

    psiticks = profile.Q(ax.get_ylim(), config.flux_units)
    Pthetamin = profile.findPtheta(psiticks.min(), units=config.canmon_units)
    Pthetamax = profile.findPtheta(psiticks.max(), units=config.canmon_units)

    ax2.set_ylabel(rf"$P_\theta's$ Fixed [{Pthetamax.units:.4g~P}]")

    num_ticks = len(ax.get_yticks())
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(nbins=num_ticks - 1))

    ax2.set_ylim([Pthetamin.m, Pthetamax.m])
    ax2.tick_params()
