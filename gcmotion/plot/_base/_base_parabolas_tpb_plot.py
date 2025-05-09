"""Simple script that draws the trapped passing boundary on the
parabolas plot"""

import numpy as np
from gcmotion.utils.logger_setup import logger

from gcmotion.entities.profile import Profile
from gcmotion.scripts.fixed_points_bif.bif_values_setup import (
    set_up_bif_plot_values,
)

from gcmotion.configuration.plot_parameters import ParabolasPlotConfig


def _plot_parabolas_tpb(
    profile: Profile,
    X_energies: list,
    O_energies: list,
    x_TPB: np.ndarray,
    ax=None,
    **kwargs,
):
    r"""Base plotting function. Only draws upon a given axis without showing
    any figures.

    Parameters
    ----------
    profile : Profile
        Profile object containing Tokamak information.
    X_energies : deque, list
        The values of the Energies of the X points for each Pzeta value.
    O_energies : deque, list
        The values of the Energies of the O points for each Pzeta value.
    ax : Axes
        The axes upon which the plot will be realized.

    Notes
    -----
    For a full list of all available optional parameters, see the dataclass
    ParabolasPlotConfig at gcmotion/configuration/plot_parameters. The
    defaults values are set there, and are overwritten if passed as
    arguements.

    """
    logger.info("\t==> Plotting Base Parabolas Trapped Passing Boundary...")

    # Unpack parameters
    config = ParabolasPlotConfig()
    for key, value in kwargs.items():
        setattr(config, key, value)

    if not profile.bfield.plain_name == "LAR":

        which_COM = "Pzeta"
        input_energy_units = "NUJoule"
        tilt_energies = False

        psip_wallNU = profile.psip_wallNU.m

        # Pzetalim is given in PzetaNU/psip_walNU so we need to multiply by
        #  psip_wallNU
        Pzetamin, Pzetamax = min(x_TPB), max(x_TPB)  # PzetaNU/psip_wallNU
        Pzetamin *= psip_wallNU  # Pzeta in NUcanmom
        Pzetamax *= psip_wallNU  # Pzeta in NUcanmom

        PzetasNU = np.linspace(Pzetamin, Pzetamax, config.TPB_density)

        # X O Energies bifurcation plot
        Pzeta_plotX, X_energies_plot = set_up_bif_plot_values(
            profile=profile,
            COM_values=PzetasNU,
            y_values=X_energies,
            which_COM=which_COM,
            tilt_energies=tilt_energies,
            input_energy_units=input_energy_units,
        )
        Pzeta_plotO, O_energies_plot = set_up_bif_plot_values(
            profile=profile,
            COM_values=PzetasNU,
            y_values=O_energies,
            which_COM=which_COM,
            tilt_energies=tilt_energies,
            input_energy_units=input_energy_units,
        )

        # The axes we will be plotting upon have y=E/(μΒ0) and x = Pζ/ψpw
        # but here we have E in NUJoule and Pζ in NUCanmom
        psip_wallNU = profile.psip_wallNU.m
        muNU = profile.muNU.m
        B0NU = profile.bfield.B0.to("NUTesla").m
        muB0NU = muNU * B0NU

        Pzeta_plotO = [
            Opoint / psip_wallNU for Opoint in Pzeta_plotO
        ]  # NUCanmom/ψpw
        Pzeta_plotX = [
            Xpoint / psip_wallNU for Xpoint in Pzeta_plotX
        ]  # NUCanmom/ψpw

        O_energies_plot = [
            Opoint / muB0NU for Opoint in O_energies_plot
        ]  # NUJoule/(μΒ0)
        X_energies_plot = [
            Xpoint / muB0NU for Xpoint in X_energies_plot
        ]  # NUJoule/(μΒ0)

        ax.scatter(
            Pzeta_plotX,
            X_energies_plot,
            s=config.TPB_X_markersize,
            color=config.TPB_X_color,
            label="X points",
        )
        ax.scatter(
            Pzeta_plotO,
            O_energies_plot,
            s=config.TPB_O_markersize,
            color=config.TPB_O_color,
            label="O points",
        )

    else:
        ax.plot(
            x_TPB,
            O_energies,
            linestyle=config.TPB_O_linestyle,
            color=config.TPB_O_color,
            label="TPB_O",
            linewidth=config.linewidth,
        )

        ax.plot(
            x_TPB,
            X_energies,
            linestyle=config.TPB_X_linestyle,
            color=config.TPB_X_color,
            label="TPB_X",
            linewidth=config.linewidth,
        )

    # In case the Pzeta limits are very close to zero set xlim accordingly
    ax.set_xlim([1.1 * config.Pzetalim[0], 1.1 * abs(config.Pzetalim[1])])
