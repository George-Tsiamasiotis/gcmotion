import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from gcmotion.configuration.plot_parameters import RZFluxContourConfig
from gcmotion.entities.profile import Profile

from gcmotion.utils.quantity_constructor import QuantityConstructor
from gcmotion.utils.logger_setup import logger


def R_Z_flux_contour(profile: Profile, **kwargs):
    r"""Plots the Magnetic Flux's contour plot in R, Z tokamak (cylindrical)
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
    flux_units : str, optional
        The units of the psi/Ptheta axis. Can be "Magnetic_flux"(same as "Tesla
        * meters^2"), "NUmagnetic_flux" (same as "NUTesla * NUmeters^2"),
        "psi_wall", "NUpsi_wall", or any other pint unit with the same
        dimensionality. Defaults to "Tesla * meters^2".
    Notes
    -----
    For a full list of all available optional parameters, see the dataclass
    RZPlotConfig at gcmotion/configuration/plot_parameters. The defaults values
    are set there, and are overwritten if passed as arguments.
    """
    logger.info("\t==> Plotting RZ Flux Contour...")

    # Unpack Parameters
    config = RZFluxContourConfig()
    for key, value in kwargs.items():
        setattr(config, key, value)

    # Create figure
    fig_kw = {
        "figsize": config.figsize,
        "dpi": config.dpi,
        "layout": config.layout,
        "facecolor": config.facecolor,
    }

    fig, ax = plt.subplots(1, 1, **fig_kw)
    logger.info("Created figure for RZ plot.")

    # Open selected dataset
    plain_name = profile.bfield.plain_name
    ds = profile.bfield.dataset
    logger.info(f"Opened dataset for {plain_name} in RZ plot.")

    # Calculate grid values for contour
    R_grid, Z_grid, Psi_grid = _get_grid_values(ds, config.parametric_density, config.flux_units)

    # Plot contour with requaested mode
    if config.mode == "lines":
        contour = ax.contour(R_grid, Z_grid, Psi_grid, levels=config.levels, cmap=config.cmap)
        logger.info("\t\tContour mode: lines")
    else:
        contour = ax.contourf(R_grid, Z_grid, Psi_grid, levels=config.levels, cmap=config.cmap)
        logger.info("\t\tContour mode: filled")

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
    ax.set_title(f"Magnetic Flux Ψ ({plain_name})", fontsize=config.title_fontsize)

    # Expand limits by adding a margin for better presentation
    x_margin = config.xmargin_perc * (R_grid.max() - R_grid.min())
    y_margin = config.ymargin_perc * (Z_grid.max() - Z_grid.min())

    ax.set_xlim(R_grid.min() - x_margin, R_grid.max() + x_margin)
    ax.set_ylim(Z_grid.min() - y_margin, Z_grid.max() + y_margin)

    # Set colorbar
    cbar = fig.colorbar(contour, cax=None, ax=ax)
    cbar.ax.set_title(label=f"Ψ [{config.flux_units}]", fontsize=config.cbarlabel_fontsize)

    plt.show()


def _get_grid_values(ds: xr.Dataset, density: int, flux_units: str) -> tuple:
    r"""Simple function that takes in a DataSet and prepares the R, Z, Ψ values
    for the RZ contour"""

    # Extract some useful quantities
    R0 = ds.raxis.data
    Z0 = ds.zaxis.data
    B0 = float(ds.Baxis.data)  # Tesla
    psi_wallNU = float(ds.psi[-1].data)  # NUMagnetic_flux

    Q = QuantityConstructor(R=R0, B0=B0, _psi_wallNU=psi_wallNU, species="p")

    _psi_values = ds.psi.data
    # We do not have measurement data at psi=0 so we add it. It is needed so
    # that there is not a void in the middle of the contour plot because
    # there was not data to interpolate in the middle (psi=0).
    _psi_values = np.insert(_psi_values, 0, 0)

    # Convert to requested flux units
    psi_values = Q(_psi_values, "NUmf").to(f"{flux_units}").m

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
    R_spline = RectBivariateSpline(theta_values, psi_values, R_values)
    Z_spline = RectBivariateSpline(theta_values, psi_values, Z_values)

    # Grid for plotting
    psi_plot = np.linspace(psi_values.min(), psi_values.max(), density)
    theta_plot = np.linspace(theta_values.min(), theta_values.max(), density)

    # Compute meshgrid
    theta_grid, psi_grid = np.meshgrid(theta_plot, psi_plot)

    # Evaluate R and Z on the grid
    R_grid = R_spline.ev(theta_grid, psi_grid)
    Z_grid = Z_spline.ev(theta_grid, psi_grid)
    Psi_grid = psi_grid  # Psi is constant on flux surfaces

    return R_grid, Z_grid, Psi_grid
