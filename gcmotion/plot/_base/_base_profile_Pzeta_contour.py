import numpy as np
from matplotlib import ticker
from matplotlib.patches import Rectangle
from matplotlib.axes import Axes
from scipy.interpolate import RectBivariateSpline

from gcmotion.utils.logger_setup import logger

from gcmotion.plot._base._config import _ProfilePzetaContourConfig

from gcmotion.entities.profile import Profile


def _base_profile_Pzeta_contour(profile: Profile, ax: Axes, **kwargs):
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
    logger.info("==> Plotting Base Profile Contour...")

    # Unpack parameters
    config = _ProfilePzetaContourConfig()
    for key, value in kwargs.items():
        setattr(config, key, value)

    # Setup meshgrid
    # The Energy calculation and y axis limits are always set with respect to
    # psilim. Pthetalim only acts as a way to move the axis window at the end,
    # or zoom to a certain area with respect to Ptheta.
    thetalim = profile.Q(config.thetalim, "radians")
    psilim = profile.Q(config.psilim, "psi_wall").to(config.flux_units)
    psigrid, thetagrid = np.meshgrid(
        np.linspace(psilim[0], psilim[1], config.grid_density),
        np.linspace(thetalim[0], thetalim[1], config.grid_density),
    )

    # Calculate Energy values
    Pzeta = profile.findPzeta(
        psigrid,
        thetagrid.m,
        config.Pzeta_units,
        potential=config.potential,
    )

    # Define data to be plotted
    # Take only the magnitudes to supress warnings
    data = {
        "theta": thetagrid.m,
        "flux": psigrid.m,
        "Pzeta": Pzeta.m,
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

    kw = {
        "cmap": config.cmap,
        "locator": locator,
        "zorder": config.zorder,
    }

    # Contour plot
    if config.mode == "lines":
        C = ax.contour("theta", "flux", "Pzeta", data=data, **kw)
    else:
        C = ax.contourf("theta", "flux", "Pzeta", data=data, **kw)

    # Setup labels.
    # Also add a second axis for Ptheta
    ax.set_ylabel(rf"$\psi$ [{psigrid.units:.4g~P}]", size=config.labelsize)
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
    if config.Pthetaax:
        ax2 = ax.twinx()
        ax2.set_ylabel(
            rf"$P_\theta$ [{psigrid.units:.4g~P}]", size=config.labelsize
        )
        psiticks = profile.Q(ax.get_yticks(), config.flux_units)
        Pthetamin = profile.findPtheta(psiticks.min())
        Pthetamax = profile.findPtheta(psiticks.max())
        ax2.yaxis.set_major_locator(ticker.MaxNLocator(config.ticknum))
        ax2.set_ylim([Pthetamin.m, Pthetamax.m])
        ax2.tick_params(labelsize=config.ticksize)

    # Add a shade above psi_wall
    if config.wall:
        rect = Rectangle(
            (
                thetalim.min().m,
                profile.Q(1, "psi_wall").to(config.flux_units).m,
            ),
            width=abs(thetalim[1].m - thetalim[0].m),
            height=psigrid[0][-1].m,
            alpha=0.2,
            color="k",
        )
        ax.add_patch(rect)

    # Format cursor
    # The format must be applied to the second ax for some reason. This means
    # we have to transform data from one ax to the other. This little manouver
    # is a bit costly but I haven't found a better way.
    cursorx = data["theta"][:, 0]
    cursory = data["flux"][0]
    cursorz = data["Pzeta"]
    values = RectBivariateSpline(cursorx, cursory, cursorz)
    flux_label = f"{psigrid.units:~P}"

    if config.Pthetaax and config.cursor:

        def cursor_format(x, y):
            # I have no idea why this works but it does :')
            Ptheta = y
            x, y = ax2.transData.transform((x, y))
            x, y = ax.transData.inverted().transform((x, y))
            z = np.take(values(x, y), 0)
            x /= np.pi
            return (
                f"θ={x:.4g}π,  "
                + f"ψ={y:.4g} {flux_label},  "
                + f"Ρθ = {Ptheta:.4g} {flux_label},  "
                + f"Pzeta = {z:.4g} {config.flux_units}"
            )

        ax2.format_coord = cursor_format
    elif not config.Pthetaax and config.cursor:

        def cursor_format(x, y):
            z = np.take(values(x, y), 0)
            x /= np.pi
            return (
                f"θ={x:.4g}π,  "
                + f"ψ={y:.4g} {flux_label},  "
                + f"Pzeta= {z:.4g} {config.flux_units}"
            )

        ax.format_coord = cursor_format

    # Return the contour object
    return C
