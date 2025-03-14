import numpy as np
from matplotlib import ticker
from matplotlib.patches import Rectangle
from matplotlib.axes import Axes
from scipy.interpolate import RectBivariateSpline

from gcmotion.utils.logger_setup import logger

from gcmotion.plot._base._config import _ProfileEnergyContourConfig

from gcmotion.entities.profile import Profile


def _base_profile_energy_contour(profile: Profile, ax: Axes, **kwargs):
    r"""Base plotting function. Only draws upon a given axis without showing
    any figures.

    Parameters
    ----------
    profile : Profile
        The profile entity.
    ax : Axes
        The ax upon which to draw.
    kwargs : dict
        The optional arguement dictionary.

    Notes
    -----
    For a full list of all available optional parameters, see the dataclass
    _ProfileContour at gcmotion/plot/_base/_config. The defaults values are set
    there, and are overwritten if passed as arguements.
    """
    logger.info("\t==> Plotting Base Profile Contour...")

    # Unpack parameters
    config = _ProfileEnergyContourConfig()
    for key, value in kwargs.items():
        setattr(config, key, value)

    # Restrict thetalim for polar projection
    if config.projection == "polar":
        config.thetalim = [-np.pi, np.pi]
        logger.debug("\t\tPolar projection: forcing thetalim to [-π,π]")

    # Setup meshgrid
    # The Energy calculation and y axis limits are always set with respect to
    # psilim. Pthetalim only acts as a way to move the axis window at the end,
    # or zoom to a certain area with respect to Ptheta.
    thetalim = profile.Q(config.thetalim, "radians")
    psilim = profile.Q(config.psilim, "psi_wall").to(config.flux_units)
    ycoordgrid, thetagrid = np.meshgrid(
        np.linspace(psilim[0], psilim[1], config.grid_density),
        np.linspace(thetalim[0], thetalim[1], config.grid_density),
    )

    # Calculate Energy values
    Energy = profile.findEnergy(
        ycoordgrid,
        thetagrid.m,
        config.E_units,
        potential=config.potential,
    )

    if config.ycoord.lower() == "ptheta":
        ycoordgrid = profile.findPtheta(ycoordgrid, "NUcanmom").m
        ycoordgrid = profile.Q(ycoordgrid, "NUcanmom").to(config.canmon_units)

    # Define data to be plotted
    # Take only the magnitudes to supress warnings
    data = {
        "theta": thetagrid.m,
        "ycoord": ycoordgrid.m,
        "Energy": Energy.m,
    }

    # =============================
    # Contour related configuration
    # =============================

    # Locator setup
    locator = (
        ticker.LogLocator(base=config.log_base, numticks=config.levels)
        if config.locator == "log"
        else ticker.MaxNLocator(nbins=config.levels)
    )
    locator.MAXTICKS = 10000
    log_msg = f"\t\tContour locator: {type(locator).__name__} "
    if config.locator == "log":
        log_msg += f"with base {config.log_base:.20g}"
    logger.debug(log_msg)

    kw = {
        "cmap": config.cmap,
        "locator": locator,
        "zorder": config.zorder,
        "linewidths": config.linewidths,
    }

    # Contour plot
    if config.mode == "lines":
        C = ax.contour(
            "theta",
            "ycoord",
            "Energy",
            data=data,
            linewidths=config.linewidths,
            **kw,
        )
        logger.debug("\t\tContour mode: lines")
    else:
        del kw["linewidths"]
        C = ax.contourf(
            "theta",
            "ycoord",
            "Energy",
            data=data,
            **kw,
        )
        logger.debug("\t\tContour mode: filled")

    # Setup labels.
    # Also add a second axis for Ptheta
    ylabel = r"\psi" if config.ycoord == "psi" else r"P_\theta"
    ax.set_ylabel(rf"${ylabel}$ [{ycoordgrid.units:.4g~P}]", size=config.labelsize)
    ax.set_xlabel(r"$\theta$ [radians]", size=config.labelsize)
    ax.tick_params(labelsize=config.ticksize)
    ax.set_xticks(
        np.linspace(-2 * np.pi, 2 * np.pi, 9),
        ["-2π", "-3π/2", "-π", "-π/2", "0", "π/2", "π", "3π/2", "2π"],
        size=config.ticksize,
    )
    ax.set_xlim(thetalim.m)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(config.ticknum))

    # Add secondary axes with Ptheta
    twin_ax_condition = bool(
        config.Pthetaax and config.projection is None and not (config.ycoord.lower() == "ptheta")
    )
    if twin_ax_condition:
        logger.debug("\t\tAdding secondary Ptheta ax.")
        ax2 = ax.twinx()
        psiticks = profile.Q(ax.get_yticks(), config.flux_units)
        Pthetamin = profile.findPtheta(psiticks.min(), units=config.canmon_units)
        Pthetamax = profile.findPtheta(psiticks.max(), units=config.canmon_units)
        ax2.set_ylabel(rf"$P_\theta$ [{Pthetamax.units:.4g~P}]", size=config.labelsize)
        ax2.yaxis.set_major_locator(ticker.MaxNLocator(config.ticknum))
        ax2.set_ylim([Pthetamin.m, Pthetamax.m])
        ax2.tick_params(labelsize=config.ticksize)
    logger.info(f"\t\tpsilim = [{psilim[0].m:.4g}, {psilim[1].m:.4g}]")

    if twin_ax_condition:
        logger.info(f"\t\tPthetalim = [{Pthetamin.m:.4g}, {Pthetamax.m:.4g}]")

    # Add a shade above psi_wall
    if config.wall and config.projection is None:
        logger.debug("\t\tAdding wall shade.")
        rect = Rectangle(
            (
                thetalim.min().m,
                profile.Q(1, "psi_wall").to(config.flux_units).m,
            ),
            width=abs(thetalim[1].m - thetalim[0].m),
            height=ycoordgrid[0][-1].m,
            alpha=0.2,
            color="k",
        )
        ax.add_patch(rect)

    ax_lim = ax.get_ylim()
    if config.wall and config.projection == "polar":
        wall_pos = profile.Q(1, "psi_wall").to(config.flux_units).m
        x = np.linspace(0, 2 * np.pi, 100)
        y_lower = np.linspace(wall_pos, wall_pos, 100)
        y_upper = np.linspace(ax_lim[1], ax_lim[1], 100)

        ax.fill_between(x=x, y1=y_lower, y2=y_upper, color="k", alpha=0.15, zorder=2)

    # Cosmetic polar projection tweaks
    if config.projection == "polar":
        ax.grid(False)
        ax.set_xticks(
            np.linspace(-np.pi, np.pi, 5),
            [" ", "3π/2", "0", "π/2", " "],
            size=config.ticksize,
        )

    # Format cursor
    # The format must be applied to the second ax for some reason. This means
    # we have to transform data from one ax to the other. This little manouver
    # is a bit costly but I haven't found a better way.
    if config.cursor:
        cursorx = data["theta"][:, 0]
        cursory = data["ycoord"][0]
        cursorz = data["Energy"]
        values = RectBivariateSpline(cursorx, cursory, cursorz)
        ycoord_label = f"{ycoordgrid.units:~P}"

    # Always add the main axes cursor, but the twin ax cursor is added only if
    # the projection is rectilinear (the default).
    if twin_ax_condition and config.cursor:
        logger.debug("\t\tAdding twin ax cursor.")

        def cursor_format(x, y):
            # I have no idea why this works but it does :')
            Ptheta = y
            x, y = ax2.transData.transform((x, y))
            x, y = ax.transData.inverted().transform((x, y))
            z = np.take(values(x, y), 0)
            x /= np.pi
            return (
                f"θ={x:.4g}π,  "
                + f"ψ={y:.4g} {ycoord_label},  "
                + f"Ρθ = {Ptheta:.4g} {ycoord_label},  "
                + f"Energy = {z:.4g} {config.E_units}"
            )

        ax2.format_coord = cursor_format
    elif not twin_ax_condition and config.cursor:
        logger.debug("\t\tAdding single ax cursor.")

        def cursor_format(x, y):
            z = np.take(values(x, y), 0)
            x /= np.pi
            return (
                f"θ={x:.4g}π,  "
                + f"ψ={y:.4g} {ycoord_label},  "
                + f"Energy = {z:.4g} {config.E_units}"
            )

        ax.format_coord = cursor_format

    ax.set_ylim(ax_lim)
    # Return the contour object
    logger.info("\t--> Contour object succesfully created and returned.")
    return C
