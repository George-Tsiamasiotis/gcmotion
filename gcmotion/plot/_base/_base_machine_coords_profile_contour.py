"""Base script that draws the selected quantity's contour plot in R, Z tokamak
(cylindrical) coordinates"""

import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from pint.errors import DimensionalityError

from matplotlib.figure import Figure
from matplotlib import ticker

from scipy.interpolate import RectBivariateSpline
from gcmotion.plot._base._config import _MachineCoordsContourConfig
from gcmotion.entities.profile import Profile
from gcmotion.entities.tokamak import Tokamak

from gcmotion.utils.logger_setup import logger


def _base_machine_coords_profile_contour(
    entity: Tokamak | Profile, fig: Figure = None, ax=None, **kwargs
):
    r"""Base plots the selected quantity's (:math:`\Psi`, E, B, I, g,
    :math:`\frac{\partial B}{\partial\theta}`, :math:`\frac{\partial
    B}{\partial\psi}`, :math:`\frac{\partial I}{\partial\psi}`,
    :math:`\frac{\partial g}{\partial\psi}`) contour plot in R, Z tokamak
    (cylindrical) coordinates.

    Parameters
    ----------
    entity : Tokamak | Profile
        The Tokamak or Profile entity. Energy contour is available only if
        entity is of type Profile.
    fig : Figure
        The figure upon which the image will be drawn. Defaults to none.
    ax : Axes
        The ax upon which the image will be drawn. Defaults to none.

    Other Parameters
    ----------------
    parametric_density : int, optional
        Practiacally the density of the :math:`\theta`, :math:`\psi` contour
        meshgrid, from which the R, Z grid is calculated. Defults to 500.
    xmargin_perc : float, optional
        x-axis margin of xlim so that there is some blank (white) space in
        between the plot limits and the contour drawing. Defaults to 0.1.
    ymargin_perc : float, optional
        y-axis margin of ylim so that there is some blank (white) space in
        between the plot limits and the contour drawing. Defaults to 0.1.
    units : str, optional
        The units of the quantity depicted on the contour. Can either be flux
        units, magnetic field units, energy units or plasma current units.
        WARNING: They must match the dimensionality of the quantity you want to
        contour.

    Notes
    -----
    For a full list of all available optional parameters, see the dataclass
    _MachineCoordsContourConfig at gcmotion/plot/_base/_config. The defaults
    values are set there, and are overwritten if passed as arguments.

    """

    # Unpack Parameters
    config = _MachineCoordsContourConfig()
    for key, value in kwargs.items():
        setattr(config, key, value)

    # Handle quantity input
    which_Q = _handle_quantity_input(config.which_Q)

    logger.info(f"\t==> Plotting RZ {which_Q} Contour...")

    # Create figure
    if fig is None:
        fig_kw = {
            "figsize": config.figsize,
            "dpi": config.dpi,
            "layout": config.layout,
            "facecolor": config.facecolor,
        }

        fig, ax = plt.subplots(1, 1, **fig_kw)
        logger.info("Created figure for RZ contour.")

    else:
        fig, ax = fig, ax

    plain_name = entity.bfield.plain_name
    logger.info(f"Opened dataset for {plain_name} in RZ contour.")

    # Calculate grid values for contour
    R_grid, Z_grid, Y_grid, Psi_grid = _get_grid_values(
        entity, which_Q, config.parametric_density, config.units
    )

    # Set locator
    locator = (
        ticker.LogLocator(base=config.log_base, numticks=config.levels)
        if config.locator == "log"
        else ticker.MaxNLocator(nbins=config.levels)
    )

    kw = {
        "cmap": config.cmap,
        "locator": locator,
        "levels": config.levels,
    }

    # Plot contour with requaested mode
    if config.mode == "lines":
        contour = ax.contour(R_grid, Z_grid, Y_grid, **kw)
        logger.info("\t\tContour mode: lines")
    else:
        contour = ax.contourf(R_grid, Z_grid, Y_grid, **kw)
        logger.info("\t\tContour mode: filled")

    # Add stationary curves if dbdtheta is plotted and if asked
    if which_Q == "b_der_theta" and config.plot_stationary_curves:
        ax.contour(
            R_grid,
            Z_grid,
            Y_grid,
            levels=[0],
            colors=config.stat_curves_color,
            linewidths=config.stat_curves_linewidth,
            linestyles=config.stat_curves_linestyle,
        )
        # Manually add legend entry
        legend_entry = Line2D(
            [0],
            [0],
            color=config.stat_curves_color,
            linewidth=config.stat_curves_linewidth,
            linestyle=config.stat_curves_linestyle,
            label=r"Stationary Curves  $\partial B / \partial \theta=0$",
        )
        ax.legend(handles=[legend_entry], loc="lower left")

    # Add black boundary around contourif asked
    if config.black_boundary:
        ax.contour(
            R_grid,
            Z_grid,
            Psi_grid,
            levels=[Psi_grid.max() * 0.999],
            colors="black",
            linewidths=config.boundary_linewidth,
        )

    # Set labels
    ax.set_xlabel("R [m]", fontsize=config.xlabel_fontsize)
    ax.set_ylabel("Z [m]", fontsize=config.ylabel_fontsize)

    # Set title
    title_Q = _get_title_format(which_Q)
    ax.set_title(f"{title_Q}", fontsize=config.title_fontsize)

    # Expand limits by adding a margin for better presentation
    x_margin = config.xmargin_perc * (R_grid.max() - R_grid.min())
    y_margin = config.ymargin_perc * (Z_grid.max() - Z_grid.min())

    ax.set_xlim(R_grid.min() - x_margin, R_grid.max() + x_margin)
    ax.set_ylim(Z_grid.min() - y_margin, Z_grid.max() + y_margin)

    # Set colorbar
    cbar = fig.colorbar(contour, cax=None, ax=ax)

    # Set colorbar ticks and fromat (All this was necessary did not work any
    # other way)
    if config.locator == "log":
        _set_up_log_locator(cbar=cbar, Y_grid=Y_grid, ticks=config.cbar_ticks)

    clabel = _get_clabel(which_Q=which_Q, units=config.units)

    cbar.ax.set_title(label=clabel, fontsize=config.cbarlabel_fontsize)


def _get_grid_values(
    entity: Tokamak | Profile, which_Q: str, density: int, units: str
) -> tuple:
    r"""Simple function that takes in a DataSet and prepares the R, Z, Y values
    for the RZ contour"""

    try:
        ds = entity.bfield.dataset
    except AttributeError:
        print(
            """WARNING: LAR MAGNETIC FIELD IS NOT NUMERICAL AND DOES NOT HAVE
            A DATASET ASSOCIATED WITH IT. USE RZ CONTOUR TO PLOT QUANTITIES
            ONLY FOR RECONSTRUCTED EQUILIBRIA WITH A DATASET THAT CORRESPONDS
            TO THEM"""
        )

        return

    # Extract some useful quantities
    R0 = ds.raxis.data
    Z0 = ds.zaxis.data

    Q = entity.Q

    _psi_valuesNU = ds.psi.data
    # We do not have measurement data at psi=0 so we add it. It is needed so
    # that there is not a void in the middle of the contour plot because
    # there was not data to interpolate in the middle (psi=0).
    _psi_valuesNU = np.insert(_psi_valuesNU, 0, 0)

    # Extract theta, R, Z data
    theta_values = ds.boozer_theta.data
    R_values = ds.R.data.T
    Z_values = ds.Z.data.T

    # Define the new column (size: (3620, 1))
    new_R_column = np.full(
        (R_values.shape[0], 1), R0
    )  # Insert R0 at first column
    new_Z_column = np.full(
        (Z_values.shape[0], 1), Z0
    )  # Insert Z0 at first column

    # Insert at the first column (axis=1), because we inserted a values 0 at
    # psi so we must insert R0 at R and Z0 at Z along a column to much shapes
    R_values = np.hstack((new_R_column, R_values))  # (3620, 101)
    Z_values = np.hstack((new_Z_column, Z_values))  # (3620, 101)

    # Interpolate
    R_spline = RectBivariateSpline(theta_values, _psi_valuesNU, R_values)
    Z_spline = RectBivariateSpline(theta_values, _psi_valuesNU, Z_values)

    # Grid for plotting
    _psi_plotNU = np.linspace(
        _psi_valuesNU.min(), _psi_valuesNU.max(), density
    )
    _theta_plot = np.linspace(theta_values.min(), theta_values.max(), density)

    # Compute meshgrid
    _theta_grid, _psi_gridNU = np.meshgrid(_theta_plot, _psi_plotNU)

    # Evaluate R and Z on the grid
    R_grid = R_spline.ev(_theta_grid, _psi_gridNU)
    Z_grid = Z_spline.ev(_theta_grid, _psi_gridNU)
    Psi_grid = _psi_gridNU  # Psi is constant on flux surfaces [NU]

    try:
        match which_Q:
            case "Ψ":
                Psi_grid = Q(Psi_grid, "NUmf").to(f"{units}").m
                Y_grid = Psi_grid
            case "Energy":
                psi_gridNU = Q(_psi_gridNU, "NUmf")
                Y_grid = entity.findEnergy(
                    psi=psi_gridNU, theta=_theta_grid, units=units
                ).m

            case "B":
                bspline = entity.bfield.b_spline
                _Y_gridNU = bspline(x=_theta_grid, y=_psi_gridNU, grid=False)
                Y_grid = Q(_Y_gridNU, "NUTesla").to(f"{units}").m

            case "I":
                ispline = entity.bfield.i_spline
                _Y_gridNU = ispline(x=_psi_gridNU)
                Y_grid = Q(_Y_gridNU, "NUpc").to(f"{units}").m

            case "g":
                gspline = entity.bfield.g_spline
                _Y_gridNU = gspline(x=_psi_gridNU)
                Y_grid = Q(_Y_gridNU, "NUpc").to(f"{units}").m

            case "b_der_theta":
                db_dtheta_spline = entity.bfield.db_dtheta_spline
                Y_grid = db_dtheta_spline(
                    x=_theta_grid, y=_psi_gridNU, grid=False
                )

            case "b_der_psi":
                db_dpsi_spline = entity.bfield.db_dpsi_spline
                Y_grid = db_dpsi_spline(
                    x=_theta_grid, y=_psi_gridNU, grid=False
                )

            case "i_der":
                ider_spline = entity.bfield.ider_spline
                Y_grid = ider_spline(x=_psi_gridNU)

            case "g_der":
                gder_spline = entity.bfield.gder_spline
                Y_grid = gder_spline(x=_psi_gridNU)

    except DimensionalityError as e:
        print(
            f"Dimensionality error encountered: {e}."
            "\n\nMAKE SURE THE UNITS YOU INPUTED DESCRIBE THE "
            "QUANTITY YOU INPUTED TO CONTOUR PLOT.\n\n"
        )
        return

    return R_grid, Z_grid, Y_grid, Psi_grid


def _handle_quantity_input(input: str) -> str:
    if input in [
        "psi",
        "Psi",
        "flux",
        "Flux",
        "magnetic flux",
        "Magnetic Flux",
        "mf",
        "magnetic_flux",
    ]:
        return "Ψ"

    if input in [
        "bfield",
        "Bfield",
        "B",
        "b",
        "magnetic field",
        "magnetic_field",
        "Magnetic Field",
        "Mf",
    ]:
        return "B"

    if input in ["energy", "Energy", "E"]:
        return "Energy"

    if input in ["i", "I", "toroidal current", "Toroidal Current"]:
        return "I"

    if input in ["g", "poloidal current", "Poloidal Current"]:
        return "g"

    if input in [
        "b_der_theta",
        "B_der_theta",
        "db/dtheta",
        "dB/dtheta",
        "dbdtheta",
        "dBdtheta",
    ]:
        return "b_der_theta"

    if input in [
        "b_der_psi",
        "B_der_psi",
        "db/dpsi",
        "dB/dpsi",
        "dbdpsi",
        "dBdpsi",
    ]:
        return "b_der_psi"

    if input in [
        "ider",
        "i_der",
        "i_der_psi",
        "didpsi",
        "dIdpsi",
        "di_dpsi",
        "dI_dpsi",
    ]:
        return "i_der"

    if input in ["gder", "g_der", "g_der_psi", "dgdpsi", "dg_dpsi"]:
        return "g_der"

    print(
        "\n\nWARNING: Selected quantity to be contoured must either be 'flux',"
        "'bfield','energy', 'ider', 'gder', 'dBdtheta', 'dBdpsi'\n\n"
    )


def _get_title_format(which_Q: str) -> str:

    d = {
        "Ψ": "Magnetic Flux 'Ψ'",
        "B": "Magnetic Field 'B'",
        "Energy": "Energy",
        "I": "Toroidal Current 'I'",
        "g": "Poloidal Current 'g'",
        "b_der_theta": r"$\partial B / \partial \theta$",
        "b_der_psi": r"$\partial B / \partial \psi$",
        "i_der": r"$\partial I / \partial \psi$",
        "g_der": r"$\partial g / \partial \psi$",
    }

    return d[which_Q]


def _set_up_log_locator(cbar, Y_grid: np.ndarray, ticks: int) -> None:
    # Define tick positions manually on a log scale
    vmin, vmax = Y_grid.min(), Y_grid.max()
    tick_positions = np.logspace(np.log10(vmin), np.log10(vmax), num=ticks)

    # Apply fixed locator to ensure only these positions are used
    cbar.ax.yaxis.set_major_locator(ticker.FixedLocator(tick_positions))

    # Use a formatter that forces all labels to appear
    cbar.ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x:.1e}")
    )


def _get_clabel(which_Q: str, units: str) -> str:
    clabel = f"{which_Q[0]} [{units}]"

    if which_Q in ["b_der_psi", "b_der_theta", "i_der", "g_der"]:
        clabel = _get_title_format(which_Q)

    return clabel
