import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from pint.errors import DimensionalityError

from scipy.interpolate import RectBivariateSpline
from gcmotion.configuration.plot_parameters import RZContourConfig
from gcmotion.entities.profile import Profile

from gcmotion.utils.logger_setup import logger


def R_Z_contour(profile: Profile, **kwargs):
    r"""Plots the selected quantity's contour plot in R, Z tokamak (cylindrical)
    coordinates.

    Parameters
    ----------
    profile : Profile
        The Profile entity.


    Other Parameters
    ----------------
    parametric_density : int, optional
        Practiacally the density of the :math:`\theta`, :math:`\psi` contour
        meshgrid. Defults to 500.
    xmargin_perc : float, optional
        x-axis margin of xlim so that there is some blank (white) space in between the
        plot limits and the contour drawing. Defaults to 0.1.
    ymargin_perc : float, optional
        y-axis margin of ylim so that there is some blank (white) space in between the
        plot limits and the contour drawing. Defaults to 0.1.
    units : str, optional
        The units of the quantity depicted on the contour. Can either be flux units,
        magnetic field units, energy units or plasma current units. WARNING: They must
        match the dimensionality of the quantity you want to contour.
    Notes
    -----
    For a full list of all available optional parameters, see the dataclass
    RZPlotConfig at gcmotion/configuration/plot_parameters. The defaults values
    are set there, and are overwritten if passed as arguments.
    """

    # Unpack Parameters
    config = RZContourConfig()
    for key, value in kwargs.items():
        setattr(config, key, value)

    # Handle quantity input
    which_Q = _handle_quantity_input(config.which_Q)

    logger.info(f"\t==> Plotting RZ {which_Q} Contour...")

    # Create figure
    fig_kw = {
        "figsize": config.figsize,
        "dpi": config.dpi,
        "layout": config.layout,
        "facecolor": config.facecolor,
    }

    fig, ax = plt.subplots(1, 1, **fig_kw)
    logger.info("Created figure for RZ contour.")

    # Open selected dataset
    plain_name = profile.bfield.plain_name
    logger.info(f"Opened dataset for {plain_name} in RZ contour.")

    # Calculate grid values for contour
    R_grid, Z_grid, Y_grid, Psi_grid = _get_grid_values(
        profile, which_Q, config.parametric_density, config.units
    )

    # Plot contour with requaested mode
    if config.mode == "lines":
        contour = ax.contour(R_grid, Z_grid, Y_grid, levels=config.levels, cmap=config.cmap)
        logger.info("\t\tContour mode: lines")
    else:
        contour = ax.contourf(R_grid, Z_grid, Y_grid, levels=config.levels, cmap=config.cmap)
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
    ax.set_title(f"{title_Q} ({plain_name})", fontsize=config.title_fontsize)

    # Expand limits by adding a margin for better presentation
    x_margin = config.xmargin_perc * (R_grid.max() - R_grid.min())
    y_margin = config.ymargin_perc * (Z_grid.max() - Z_grid.min())

    ax.set_xlim(R_grid.min() - x_margin, R_grid.max() + x_margin)
    ax.set_ylim(Z_grid.min() - y_margin, Z_grid.max() + y_margin)

    # Set colorbar
    cbar = fig.colorbar(contour, cax=None, ax=ax)
    clabel = f"{which_Q[0]} [{config.units}]"

    if which_Q in ["b_der_psi", "b_der_theta", "i_der", "g_der"]:
        clabel = _get_title_format(which_Q)

    cbar.ax.set_title(label=clabel, fontsize=config.cbarlabel_fontsize)

    plt.show()


def _get_grid_values(profile: Profile, which_Q: str, density: int, units: str) -> tuple:
    r"""Simple function that takes in a DataSet and prepares the R, Z, Y values
    for the RZ contour"""

    ds = profile.bfield.dataset

    # Extract some useful quantities
    R0 = ds.raxis.data
    Z0 = ds.zaxis.data

    Q = profile.Q

    _psi_valuesNU = ds.psi.data
    # We do not have measurement data at psi=0 so we add it. It is needed so
    # that there is not a void in the middle of the contour plot because
    # there was not data to interpolate in the middle (psi=0).
    _psi_valuesNU = np.insert(_psi_valuesNU, 0, 0)

    # Convert to requested flux units
    psi_valuesNU = Q(_psi_valuesNU, "NUmf")

    # Extract theta, R, Z data
    theta_values = ds.boozer_theta.data
    R_values = ds.R.data.T
    Z_values = ds.Z.data.T

    # Define the new column (size: (3620, 1))
    new_R_column = np.full((R_values.shape[0], 1), R0)  # Insert R0 at first column
    new_Z_column = np.full((Z_values.shape[0], 1), Z0)  # Insert Z0 at first column

    # Insert at the first column (axis=1), because we inserted a values 0 at psi
    # so we must insert R0 at R and Z0 at Z along a column to much shapes
    R_values = np.hstack((new_R_column, R_values))  # (3620, 101)
    Z_values = np.hstack((new_Z_column, Z_values))  # (3620, 101)

    # Interpolate
    R_spline = RectBivariateSpline(theta_values, psi_valuesNU.m, R_values)
    Z_spline = RectBivariateSpline(theta_values, psi_valuesNU.m, Z_values)

    # Grid for plotting
    _psi_plotNU = np.linspace(psi_valuesNU.m.min(), psi_valuesNU.m.max(), density)
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
            # WHEN PULL FROM GEORGE DO NOT PUT INPUT QUANTITY NECESSARILY
            case "Energy":
                psi_gridNU = Q(_psi_gridNU, "NUMagnetic_flux")
                Y_grid = profile.findEnergy(psi=psi_gridNU, theta=_theta_grid, units=units).m

            case "B":
                bspline = profile.bfield.b_spline
                _Y_gridNU = bspline(x=_theta_grid, y=_psi_gridNU, grid=False)
                Y_grid = Q(_Y_gridNU, "NUTesla").to(f"{units}").m

            case "I":
                ispline = profile.bfield.i_spline
                _Y_gridNU = ispline(x=_psi_gridNU)
                Y_grid = Q(_Y_gridNU, "NUpc").to(f"{units}").m

            case "g":
                gspline = profile.bfield.g_spline
                _Y_gridNU = gspline(x=_psi_gridNU)
                Y_grid = Q(_Y_gridNU, "NUpc").to(f"{units}").m

            case "b_der_theta":
                db_dtheta_spline = profile.bfield.db_dtheta_spline
                Y_grid = db_dtheta_spline(x=_theta_grid, y=_psi_gridNU, grid=False)

            case "b_der_psi":
                db_dpsi_spline = profile.bfield.db_dpsi_spline
                Y_grid = db_dpsi_spline(x=_theta_grid, y=_psi_gridNU, grid=False)

            case "i_der":
                ider_spline = profile.bfield.ider_spline
                Y_grid = ider_spline(x=_psi_gridNU)

            case "g_der":
                gder_spline = profile.bfield.gder_spline
                Y_grid = gder_spline(x=_psi_gridNU)

    except DimensionalityError as e:
        print(
            f"Dimensionality error encountered: {e}.\n\nMAKE SURE THE UNITS YOU INPUTED DESCRIBE THE QUANTITY YOU INPUTED TO CONTOUR PLOT.\n\n"
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

    if input in ["b_der_theta", "B_der_theta", "db/dtheta", "dB/dtheta", "dbdtheta", "dBdtheta"]:
        return "b_der_theta"

    if input in ["b_der_psi", "B_der_psi", "db/dpsi", "dB/dpsi", "dbdpsi", "dBdpsi"]:
        return "b_der_psi"

    if input in ["ider", "i_der", "i_der_psi", "didpsi", "dIdpsi", "di_dpsi", "dI_dpsi"]:
        return "i_der"

    if input in ["gder", "g_der", "g_der_psi", "dgdpsi", "dg_dpsi"]:
        return "g_der"

    print(
        "\n\nWARNING: Selected quantity to be contoured must either be 'flux', 'bfield','energy', 'ider', 'gder', 'dBdtheta', 'dBdpsi'\n\n"
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
