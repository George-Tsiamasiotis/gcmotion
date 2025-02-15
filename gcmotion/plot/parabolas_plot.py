"""
Simple script that draws parabolas diagram along with the trapped passing LAR boundary
if asked.
"""

from gcmotion.utils.logger_setup import logger
import matplotlib.pyplot as plt

from gcmotion.configuration.plot_parameters import ParabolasPlotConfig
from gcmotion.entities.profile import Profile
from gcmotion.plot._base._base_calc_parabolas import _calc_parabolas


def parabolas_diagram(profile: Profile, ax=None, **kwargs):
    r"""

    This script draw the parabolas diagram along with the trapped passing LAR boundary
    (if asked) by plotting the values calculated in :py:func:`_calc_parabolas`.

    Parameters
    ----------
    profile : Profile
        Profile object that contains Tokamak information like bfield, mu,
        useful psi values.
    ax : Axes, optional
        Axes object upon which the plot is to be created. Defaults to None.

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
    LAR_TPB : bool, optional
        Boolean that determines weather the LAR trapped-passing boundary is to
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

    # Create figure if ax is not given
    if ax is None:
        fig_kw = {
            "figsize": config.figsize,
            "dpi": config.dpi,
            "layout": config.layout,
            "facecolor": config.facecolor,
        }

        fig, par_ax = plt.subplots(1, 1, **fig_kw)

        logger.info("Axes was not given for parabolas. Creating figure")
    else:
        par_ax = ax
        logger.info("Using Axes given to parabolas diagram")

    # Calculate parabolas values
    x, x_LAR, y_R, y_L, y_MA, LAR_tpb_O, LAR_tpb_X, v_R, v_L, v_MA = _calc_parabolas(
        Pzetalim=config.Pzetalim, profile=profile, Pzeta_density=config.Pzeta_density
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

    # If asked and if LAR plot LAR trapped-passing boundary
    if config.LAR_TPB and profile.bfield.plain_name == "LAR":
        par_ax.plot(
            x_LAR,
            LAR_tpb_O,
            linestyle=config.TPB_O_linestyle,
            color=config.TPB_O_color,
            label="TPB_LAR_O",
            linewidth=config.linewidth,
        )
        par_ax.plot(
            x_LAR,
            LAR_tpb_X,
            linestyle=config.TPB_X_linestyle,
            color=config.TPB_X_color,
            label="TPB_LAR_X",
            linewidth=config.linewidth,
        )
        # In case the Pzeta limits are very close to zero
        par_ax.set_xlim([1.1 * config.Pzetalim[0], 1.1 * abs(config.Pzetalim[1])])

        logger.info("Plotted LAR trapped-passing boundary in parabolas diagram.")

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
