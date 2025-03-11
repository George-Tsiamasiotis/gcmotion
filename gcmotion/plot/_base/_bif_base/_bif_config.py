r"""
Sets up configuration for base plotting scripts.

All these values can be overwritten if passed as an arguement to the
corresponding function.
"""

from dataclasses import dataclass


@dataclass
class _ThetasFixedPlotConfig:
    # Thetas of X points keywords
    thetas_X_marker: str = "."
    thetas_X_markersize: float = 2
    thetas_X_markercolor: str = "#E65100"
    # Thetas of O points keywords
    thetas_O_marker: str = "."
    thetas_O_markersize: float = 2
    thetas_O_markercolor: str = "#1f77b4"
    # Label keywords
    theta_ylabel_fontsize: float = 10
    theta_ylabel_rotation: int = 90
    # Legend keywords
    theta_legend: bool = False


@dataclass
class _PsisFixedPlotConfig:
    # psis of X points keywords
    psis_X_marker: str = "."
    psis_X_markersize: float = 2
    psis_X_markercolor: str = "#E65100"
    # psis of O points keywords
    psis_O_marker: str = "."
    psis_O_markersize: float = 2
    psis_O_markercolor: str = "#1f77b4"
    # Label keywords
    psi_ylabel_rotation: int = 90
    psi_ylabel_fontsize: float = 10
    # Legend keywords
    psi_legend: bool = True
    psi_legend_loc: str = "lower left"
    # Wall dashed line limit keywords
    wall_line_color: str = "black"
    wall_linestyle: str = "dashed"
    wall_linewidth: int = 1
    wall_line_alpha: float = 0.5
    flux_units: str = "NUmf"
    canmon_units: str = "NUcanmom"


@dataclass
class _NDFPlotConfig:
    # Number of X points keywords
    ndfp_X_marker: str = "."
    ndfp_X_markersize: float = 2
    ndfp_X_markercolor: str = "#E65100"
    # Number of O points keywords
    ndfp_O_marker: str = "."
    ndfp_O_markersize: float = 2
    ndfp_O_markercolor: str = "#1f77b4"
    # Label keywords
    ndfp_ylabel_fontsize: float = 10
    ndfp_ylabel_rotation: int = 90
    # Legend keywords
    ndfp_legend: bool = True


@dataclass
class _TPBPlotConfig:
    # Figure keywords
    figsize: tuple = (13, 7)
    dpi: int = 100
    layout: str = "constrained"
    facecolor: str = "white"
    # Energies of X points keywords
    tpb_X_marker: str = "."
    tpb_X_markersize: float = 2
    tpb_X_markercolor: str = "#E65100"
    # Energies of O points keywords
    tpb_O_marker: str = "."
    tpb_O_markersize: float = 2
    tpb_O_markercolor: str = "#1f77b4"
    # Title keywords
    tpb_title_fontsize: float = 15
    tpb_title_color: str = "black"
    # Labels keywords
    tpb_ylabel_fontzise: float = 10
    tpb_xlabel_fontzise: float = 13
    tpb_ylabel_rotation: int = 90
    # Legend keywords
    tpb_legend: bool = True
    # Units keywords
    energy_units: str = "NUJoule"
