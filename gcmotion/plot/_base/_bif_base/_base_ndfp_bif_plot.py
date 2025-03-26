r"""Simple script that draws the number of distinct fixed points for each
:math:`\mu` or :math:`P_{\zeta}` of the bifurcation analysis"""

from matplotlib.ticker import MaxNLocator
from gcmotion.utils.logger_setup import logger

from gcmotion.plot._base._bif_base._bif_config import _NDFPlotConfig


def _ndfp_bif_plot(
    COM_values: list,
    num_of_XP: list,
    num_of_OP: list,
    ax=None,
    **kwargs,
):
    r"""Base plotting function. Only draws upon a given axis without showing
    any figures.

    Parameters
    ----------
    COM_values : list, deque
        List of COM values :math:`P_{\zeta}`'s or :math:`\mu`'s in [NU].
    num_of_XP : deque, list
        The number of distinct X points found for each COM value.
    num_of_OP : deque, list
        The number of distinct O points found for each COM value.
    which_COM : str
        String that indicates with respect to which constant of motion (COM)
        :math:`\mu` or :math:`P_{\zeta}` the number of distinct fixed points
        fixed are plotted.
    ax : Axes
        The ax upon which to draw.

    Notes
    -----
    For a full list of all available optional parameters, see the dataclass
    _NDFPlotConfig at gcmotion/plot/_base/_bif_base/_bif_config. The defaults
    values are set there, and are overwritten if passed as arguements.

    """
    logger.info(
        """\t==> Plotting Base Number of distinct Fixed Points Bifurcation
        Diagram..."""
    )

    # Unpack parameters
    config = _NDFPlotConfig()
    for key, value in kwargs.items():
        setattr(config, key, value)

    # Number of distinct fixed points Diagram
    selected_COMs = COM_values
    ax.set_ylabel(
        "Number of Fixed Points",
        rotation=config.ndfp_ylabel_rotation,
        fontsize=config.ndfp_ylabel_fontsize,
    )
    ax.scatter(
        selected_COMs,
        num_of_XP,
        marker=config.ndfp_X_marker,
        s=config.ndfp_X_markersize,
        color=config.ndfp_X_markercolor,
        label="X points",
    )
    ax.scatter(
        selected_COMs,
        num_of_OP,
        marker=config.ndfp_O_marker,
        s=config.ndfp_O_markersize,
        color=config.ndfp_O_markercolor,
        label="O points",
    )

    # If multiple unique y-values exist, use MaxNLocator
    if len(set(num_of_XP)) > 1 or len(set(num_of_OP)) > 1:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    else:
        # If only a single unique y-value, set the tick manually
        ax.set_yticks([int(num_of_OP[0])])

    if config.ndfp_legend:
        ax.legend()
