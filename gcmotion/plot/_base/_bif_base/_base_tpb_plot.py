r"""Simple script that draws the energy of the distinct fixed points
for each :math:`\mu` or :math:`P_{\zeta}` of the bifurcation analysis"""

from gcmotion.utils.logger_setup import logger
import matplotlib.pyplot as plt
from gcmotion.entities.profile import Profile

from gcmotion.scripts.fixed_points_bif.bif_values_setup import (
    set_up_bif_plot_values,
)
from gcmotion.plot._base._bif_base._bif_config import _TPBPlotConfig


def _plot_trapped_passing_boundary(
    profile: Profile,
    COM_values: list,
    X_energies: list,
    O_energies: list,
    which_COM: str,
    input_energy_units: str,
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
    X_energies : deque, list
        The values of the Energies of the X points for each COM value.
    O_energies : deque, list
        The values of the Energies of the O points for each COM value.
    which_COM : str
        String that indicates with respect to which constant of motion (COM)
        :math:`\mu` or :math:`P_{\zeta}` the energies of the fixed points are
        plotted.

    Notes
    -----
    For a full list of all available optional parameters, see the dataclass
    _TPBPlotConfig at gcmotion/plot/_base/_bif_base/_bif_config. The defaults
    values are set there, and are overwritten if passed as arguements.


    """
    logger.info("\t==> Plotting Base Trapped Passing Boundary...")

    # Unpack parameters
    config = _TPBPlotConfig()
    for key, value in kwargs.items():
        setattr(config, key, value)

    if ax is None:
        fig_kw = {
            "figsize": config.figsize,
            "dpi": config.dpi,
            "layout": config.layout,
            "facecolor": config.facecolor,
        }

        fig, ax = plt.subplots(1, 1, **fig_kw)

    # If the selcted COM is mu we need to tilt the energies in the plot by
    # subtracting mu*B0 which is done in set_up_bif_plot_values if
    # tilt_energies == True
    if which_COM == "mu":
        tilted_energies_loc = True
    elif which_COM == "Pzeta":
        tilted_energies_loc = False

    # X O Energies bifurcation plot
    COM_plotX, X_energies_plot = set_up_bif_plot_values(
        profile=profile,
        COM_values=COM_values,
        y_values=X_energies,
        which_COM=which_COM,
        tilt_energies=tilted_energies_loc,
        input_energy_units=input_energy_units,
    )
    COM_plotO, O_energies_plot = set_up_bif_plot_values(
        profile=profile,
        COM_values=COM_values,
        y_values=O_energies,
        which_COM=which_COM,
        tilt_energies=tilted_energies_loc,
        input_energy_units=input_energy_units,
    )

    set_up_dict = _set_up_tpb_base_plot(
        profile=profile,
        COM_plotO=COM_plotO,
        O_energies_plot=O_energies_plot,
        COM_plotX=COM_plotX,
        X_energies_plot=X_energies_plot,
        which_COM=which_COM,
        label_energy_units=input_energy_units,
    )

    x_label_loc = set_up_dict["x_label_loc"]
    y_label_loc = set_up_dict["y_label_loc"]
    xX_values = set_up_dict["xX_values"]
    yX_values = set_up_dict["yX_values"]
    xO_values = set_up_dict["xO_values"]
    yO_values = set_up_dict["yO_values"]

    ax.set_title(
        rf"Bifurcation Diagram ({profile.bfield.plain_name})",
        fontsize=config.tpb_title_fontsize,
        color=config.tpb_title_color,
    )

    ax.set_xlabel(x_label_loc, fontsize=config.tpb_xlabel_fontzise)
    ax.set_ylabel(
        y_label_loc,
        rotation=config.tpb_ylabel_rotation,
        fontsize=config.tpb_ylabel_fontzise,
    )
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    ax.scatter(
        xX_values,
        yX_values,
        marker=config.tpb_X_marker,
        s=config.tpb_X_markersize,
        color=config.tpb_X_markercolor,
        label="X points",
    )
    ax.scatter(
        xO_values,
        yO_values,
        marker=config.tpb_O_marker,
        s=config.tpb_O_markersize,
        color=config.tpb_O_markercolor,
        label="O points",
    )

    if config.tpb_legend:
        ax.legend()


def _set_up_tpb_base_plot(
    profile: Profile,
    COM_plotO: list,
    O_energies_plot: list,
    COM_plotX: list,
    X_energies_plot: list,
    which_COM: str,
    label_energy_units: str,
):
    r"""
    Simple script that sets up the base trapped passing boundary plot
    by checking some parameters.
    """

    if which_COM == "mu":
        x_label_loc = r"$E-{\mu}B_0$" + f"[{label_energy_units}]"
        y_label_loc = r"$\mu$" + f"[{profile.muNU.units}]"
        xO_values = O_energies_plot
        yO_values = COM_plotO
        xX_values = X_energies_plot
        yX_values = COM_plotX

    elif which_COM == "Pzeta":
        x_label_loc = r"$P_{\zeta}$" + f"[{profile.PzetaNU.units}]"
        y_label_loc = f"Energies [{label_energy_units}]"
        xO_values = COM_plotO
        yO_values = O_energies_plot
        xX_values = COM_plotX
        yX_values = X_energies_plot

    else:
        print(
            """\n'which_COM' arguments must be either 'mu' or 'Pzeta'.
            \nABORTING trapped passing boundary plot...\n"""
        )

    return {
        "x_label_loc": x_label_loc,
        "y_label_loc": y_label_loc,
        "xX_values": xX_values,
        "yX_values": yX_values,
        "xO_values": xO_values,
        "yO_values": yO_values,
    }
