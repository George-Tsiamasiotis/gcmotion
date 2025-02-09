r"""
Magnetic Profile
----------------

Plots a poloidal cut of the magnetic field strength and plasma current
intensities.
"""

import pint
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

from gcmotion.utils.logger_setup import logger
from gcmotion.entities.tokamak import Tokamak
from gcmotion.entities.profile import Profile
from gcmotion.entities.particle import Particle

from gcmotion.configuration.plot_parameters import MagneticProfileConfig

# Quantity alias for type annotations
type Quantity = pint.UnitRegistry.Quantity


def magnetic_profile(entity: Tokamak | Profile | Particle, **kwargs):
    r"""Plots a poloidal cut of the magnetic field strength and plasma
    current intensities.

    Parameters
    ----------
    entity : Tokamak, Profile, Particle
        The object to plot the qfactor of.

    Other Parameters
    ----------------
    span : list, optional
        The x-axis span, relative to :math:`\psi_{wall}`. Defaults to [0, 1.1].
    units: {"NU", "SI"}, optional
        The Quantities' units. Defaults to "NU".
    coord: {"psi", "r"}, optional
        Which variable to use as the polar coordinate. Defaults to "r"
    grid_density: int, optional
        The contour plots' grid density. Defaults to 100.
    levels: int, optional
        The contour plots' levels. Defaults to 20.

    """
    logger.info("==> Plotting Magnetic field profile...")

    # Unpack parameters
    config = MagneticProfileConfig()
    for key, value in kwargs.items():
        setattr(config, key, value)

    # Grab needed objects and needed attributes
    Q = entity.Q
    bfield = entity.bfield
    psi_wallNU = entity.psi_wallNU
    if hasattr(bfield, "is_numerical") and config.span[1] > 1:
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
    ax_dict = fig.subplot_mosaic(
        [
            ["b", "i_polar", "g_polar"],
            [".", "i_der2d", "g_der2d"],
        ],
        per_subplot_kw={
            "b": {"projection": "polar"},
            "i_polar": {"projection": "polar"},
            "g_polar": {"projection": "polar"},
        },
    )
    fig.suptitle(f"Magnetic Field Profile ({bfield.plain_name})")
    axb, axi, axg = ax_dict["b"], ax_dict["i_polar"], ax_dict["g_polar"]
    axid, axgd = ax_dict["i_der2d"], ax_dict["g_der2d"]

    # Grid creation and (b,i,g) calculation
    # Coordinates psi, r, theta are all in [NU]
    psi, theta = np.meshgrid(
        np.linspace(0, psi_wallNU.m, config.grid_density),
        np.linspace(0, 2 * np.pi, config.grid_density),
    )
    b, i, g = bfield.bigNU(psi, theta)
    if config.coord == "psi":
        coord = Q(psi, "NUMagnetic_flux")
        logger.info("\tPlotting with respect to psi.")
    else:
        coord = Q(np.sqrt(2 * psi), "NUmeters")
        logger.info("\tPlotting with respect to r.")

    # Set appropriate units:
    b = Q(b, "NUTesla")
    i = Q(i, "NUPlasma_current")
    g = Q(g, "NUPlasma_current")
    log_units = "NU"
    if config.units.lower() == "si":
        b.ito("Tesla")
        i.ito("Plasma_current")
        g.ito("Plasma_current")
        if config.coord == "psi":
            coord.ito("Magnetic_flux")
        else:
            coord.ito("meters")
        log_units = "SI"
    logger.info(f"\tPlotting in {log_units}.")

    # Locator setup
    locator = (
        ticker.LogLocator(base=config.log_base, numticks=config.levels)
        if config.locator == "log"
        else ticker.MaxNLocator(nbins=config.levels)
    )
    log_msg = f"\tContour locator: {type(locator).__name__} "
    if config.locator == "log":
        log_msg += f"with base {config.log_base:.20g}"
    logger.debug(log_msg)

    # ===============================================================

    # Bfield contour
    bcontour_kw = {
        "levels": config.levels,
        "cmap": config.bcmap,
        "locator": locator,
    }
    Cb = axb.contourf(theta, coord.m, b.m, **bcontour_kw)
    fig.colorbar(
        Cb,
        ax=axb,
        shrink=0.8,
        spacing="proportional",
        label=f"[{b.units:~P}]",
    )

    # I contour
    icontour_kw = {
        "levels": config.levels,
        "cmap": config.icmap,
    }
    C1 = axi.contourf(theta, coord.m, i.m, **icontour_kw)
    fig.colorbar(
        C1,
        ax=axi,
        shrink=0.8,
        label=f"[{i.units:~P}]",
    )

    # g contour
    gcontour_kw = {
        "levels": config.levels,
        "cmap": config.gcmap,
    }
    C2 = axg.contourf(theta, coord.m, g.m, **gcontour_kw)
    fig.colorbar(
        C2,
        ax=axg,
        shrink=0.8,
        label=f"[{g.units:~P}]",
    )

    # ===============================================================

    axid.plot(
        psi[0] / psi_wallNU.m,
        i[0],
        c=config.current_color,
        lw=config.linewidth,
    )
    axgd.plot(
        psi[0] / psi_wallNU.m,
        g[0],
        c=config.current_color,
        lw=config.linewidth,
    )

    axid.set_title("Toroidal Current 'I'")
    axgd.set_title("Poloidal Current 'g'")
    axid.set_ylabel(f"I [{i.units:~P}]", c=config.current_color)
    axgd.set_ylabel(f"g [{g.units:~P}]", c=config.current_color)
    axid.tick_params(axis="y", labelcolor=config.current_color)
    axgd.tick_params(axis="y", labelcolor=config.current_color)

    if config.coord == "psi":
        axid.set_xlabel(r"$\psi/\psi_{wall}$", size=config.labelsize)
        axgd.set_xlabel(r"$\psi/\psi_{wall}$", size=config.labelsize)
    else:
        axid.set_xlabel(r"$r/r_{wall}$", size=config.labelsize)
        axgd.set_xlabel(r"$r/r_{wall}$", size=config.labelsize)

    # If the magnetic field is numerical, also plot the derivatives
    if getattr(bfield, "is_numerical", False) and config.plot_derivatives:
        logger.debug("\tNumerical bfield: Plotting currents' derivatives.")
        ider = bfield.ider_spline(psi[0])
        gder = bfield.gder_spline(psi[0])
        axid2 = axid.twinx()
        axgd2 = axgd.twinx()
        wall_norm = (
            psi_wallNU.m
            if config.coord == "psi"
            else np.sqrt(2 * psi_wallNU.m)
        )
        axid2.plot(
            coord[0] / wall_norm,
            ider,
            c=config.derivative_color,
            lw=config.linewidth,
        )
        axgd2.plot(
            coord[0] / wall_norm,
            gder,
            c=config.derivative_color,
            lw=config.linewidth,
        )

        ider_label = r"$\partial I / \partial \psi$"
        gder_label = r"$\partial g / \partial \psi$"
        axid2.set_ylabel(
            ider_label, c=config.derivative_color, size=config.labelsize
        )
        axgd2.set_ylabel(
            gder_label, c=config.derivative_color, size=config.labelsize
        )
        axid2.tick_params(axis="y", labelcolor=config.derivative_color)
        axgd2.tick_params(axis="y", labelcolor=config.derivative_color)

    axid.margins(0)
    axgd.margins(0)

    # ===============================================================

    # Misc plotting
    axb.set_title(
        f"Magnetic field strength [{b.units:~P}]", pad=config.ax_title_pad
    )
    axi.set_title(
        rf"Toroidal current 'I'[{i.units:~P}]", pad=config.ax_title_pad
    )
    axg.set_title(
        rf"Poloidal current 'g'[{g.units:~P}]", pad=config.ax_title_pad
    )

    axb.set_rlabel_position(70)
    axi.set_rlabel_position(70)
    axg.set_rlabel_position(70)

    xlabels = ("0", "π/2", "π", "3π/2")
    axb.set_xticks(np.linspace(0, 3 / 2 * np.pi, 4), xlabels)
    axi.set_xticks(np.linspace(0, 3 / 2 * np.pi, 4), xlabels)
    axg.set_xticks(np.linspace(0, 3 / 2 * np.pi, 4), xlabels)

    coord_max = coord[0, -1]
    axb.set_yticks([coord_max.m], labels=[f"{coord_max:.3g}"])
    axi.set_yticks([coord_max.m], labels=[f"{coord_max:.3g}"])
    axg.set_yticks([coord_max.m], labels=[f"{coord_max:.3g}"])

    axb.grid(alpha=0.2)
    axi.grid(alpha=0.2)
    axg.grid(alpha=0.2)

    if config.show:
        logger.info("--> Magnetic profile successfully plotted.")
        plt.show()
    else:
        logger.info("--> Magnetic profile returned without plotting")
        plt.clf()
