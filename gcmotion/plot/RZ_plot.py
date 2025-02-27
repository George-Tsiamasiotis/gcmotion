import numpy as np
import xarray as xr
import os
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from gcmotion.entities.profile import Profile


def R_Z_plot(
    density: int,
    eqdsk_filename: str = "dtt_negative.nc",
    xmargin_perc: float = 0.1,
    ymargin_perc: float = 0.1,
):

    grandparent_dir = os.path.dirname(os.path.dirname(__file__))
    # Define subdirectory path and filename
    target_dir = os.path.join(grandparent_dir, "tokamak", "reconstructed")
    full_path = os.path.join(target_dir, eqdsk_filename)
    # Create figure
    fig_kw = {
        "figsize": (7, 13),
        "dpi": 100,
        "layout": "constrained",
        "facecolor": "lightskyblue",
    }

    fig, ax = plt.subplots(1, 1, **fig_kw)

    ds = xr.open_dataset(full_path)

    psi_values = ds.psi.data
    psi_values = np.insert(psi_values, 0, 0)

    theta_values = ds.boozer_theta.data

    R_values = ds.R.data.T
    Z_values = ds.Z.data.T

    R0 = ds.raxis.data
    Z0 = ds.zaxis.data

    # Define the new column (size: (3620, 1))
    new_R_column = np.full((R_values.shape[0], 1), R0)  # Insert R0 at first column
    new_Z_column = np.full((Z_values.shape[0], 1), Z0)  # Insert Z0 at first column

    # Insert at the first column (axis=1)
    R_values = np.hstack((new_R_column, R_values))  # (3620, 101)
    Z_values = np.hstack((new_Z_column, Z_values))  # (3620, 101)

    R_spline = RectBivariateSpline(theta_values, psi_values, R_values)
    Z_spline = RectBivariateSpline(theta_values, psi_values, Z_values)

    # Grid for plotting
    psi_plot = np.linspace(psi_values.min(), psi_values.max(), density)
    theta_plot = np.linspace(0, 2 * np.pi, density)

    # Compute meshgrid
    theta_grid, psi_grid = np.meshgrid(theta_plot, psi_plot)

    # Evaluate R and Z on the grid
    R_grid = R_spline.ev(theta_grid, psi_grid)
    Z_grid = Z_spline.ev(theta_grid, psi_grid)
    Psi_grid = psi_grid  # Psi is constant on flux surfaces

    # Plot contour
    contour = ax.contourf(R_grid, Z_grid, Psi_grid, levels=20, cmap="plasma")
    # **Add black boundary**
    ax.contour(
        R_grid, Z_grid, Psi_grid, levels=[Psi_grid.max() * 0.999], colors="black", linewidths=2
    )

    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title("Magnetic Flux Contours (Ψ)")

    # Expand limits by adding a margin
    x_margin = xmargin_perc * (R_grid.max() - R_grid.min())
    y_margin = ymargin_perc * (Z_grid.max() - Z_grid.min())

    ax.set_xlim(R_grid.min() - x_margin, R_grid.max() + x_margin)
    ax.set_ylim(Z_grid.min() - y_margin, Z_grid.max() + y_margin)
    fig.colorbar(contour, label="Ψ")

    plt.show()
