r"""
Sets up configuration for base plotting scripts.

All these values can be overwritten if passed as an arguement to the
corresponding function.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class _ProfileEnergyContourConfig:
    # Default optional arguements
    thetalim: tuple = (-np.pi, np.pi)
    psilim: tuple = (0, 1.2)
    levels: int = 30
    ycoord: str = "psi"  # "psi" or "Ptheta"
    flux_units: str = "NUMagnetic_flux"
    canmon_units: str = "NUCanonical_momentum"
    E_units: str = "keV"
    potential: bool = True
    wall: bool = True
    # Contour
    projection: str | None = None  # None = default
    mode: str = "filled"  # "filled" or "lines"
    grid_density: int = 200
    cmap: str = "plasma"
    locator: str = "log"
    log_base: float = 1.0001
    zorder: int = 0
    bold_lines: bool = True
    linewidths: float = 1
    linecolors: str = "black"
    plot_level_lines: bool = True
    separatrix_linewidth: float = 4
    separatrix_linecolor: str = "red"
    Pthetaax: bool = True
    cursor: bool = True
    # Labels
    labelsize: float = 20
    labelsize_Ptheta: float = 15
    ticknum: int = 10
    # Ticks
    x_ticksize: float = 20
    y_ticksize: float = 20
    y_Ptheta_ticksize: float = 15
    # Colorbar
    numticks: int = 10
    cbarlabelsize: int = 12
    # Polar Projection
    polar_ytick_num: int = 5
    polar_ytick_size: float = 12


@dataclass
class _ColorbarConfig:
    location: str = "top"
    numticks: int = 15
    label: str = ""
    labelsize: float = 15
    # Energy line
    energy_line: None = None
    energy_line_color: str = "r"
    energy_line_style: str = "-"
    energy_line_zorder: int = 3


@dataclass
class _ParticlePoloidalDrift:
    units: str = "SI"  # "SI" or "NU"
    thetalim: tuple = (-np.pi, np.pi)
    percentage: int = 100
    flux_units: str = "Tesla * meter^2"
    initial: bool = True
    # Scatter kw
    s: float = 1
    color: str = "r"
    marker: str = "."
    # Inital point
    init_s: float = 30
    init_color: str = "k"
    init_marker: str = "."
    # Ticks
    x_ticksize: float = 20
    y_ticksize: float = 20


@dataclass
class _MachineCoordsContourConfig:
    # Figure keywords
    figsize: tuple = (6, 8)
    dpi: int = 100
    layout: str = "constrained"
    facecolor: str = "white"
    # Contour keywords
    cmap: str = "managua"  # "BrBG, viridis, managua"
    levels: int = 25
    mode: str = None
    units: str = "NUmf"
    which_Q: str = "flux"
    plot_parabolas: bool = True
    # Locator keywords
    log_base: float = 1.0001
    locator: str = "linear"
    # Boundary keywords
    black_boundary: bool = True
    boundary_linewidth: int = 1
    # Stationary curves keywords
    plot_stationary_curves: bool = True
    stat_curves_color: str = "black"
    stat_curves_linewidth: float = 1
    stat_curves_linestyle: str = "dashed"
    # Labels - Title keywords
    xlabel_fontsize: float = 20
    ylabel_fontsize: float = 20
    title_fontsize: float = 15
    # Colorbar keywords
    cbarlabel_fontsize: float = 10
    cbar_ticks: int = 10
    # Numerical keywords
    parametric_density: int = 500
    xmargin_perc: float = 0.1
    ymargin_perc: float = 0.1
    # Ticks
    x_ticksize: float = 20
    y_ticksize: float = 20


@dataclass()
class _FixedPointsPlotConfig:
    # Figure keywords
    figsize: tuple = (10, 7)
    dpi: int = 100
    layout: str = "constrained"
    facecolor: str = "white"
    projection: str | None = None  # None = default
    # Fixed points keywords
    fp_plot_init_cond: bool = False
    flux_units: str = "NUMagnetic_flux"
    X_color: str = "#80FF80"
    O_color: str = "yellow"
    X_size: float = 100
    O_size: float = 100
    # IC Keywords
    ic_marker: str = ">"
    ic_markercolor: str = "red"
    ic_markersize: float = 100
    # RZ coords
    RZ_coords: bool = False
    separatrices: bool = False
