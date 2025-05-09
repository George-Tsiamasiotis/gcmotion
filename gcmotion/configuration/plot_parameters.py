from gcmotion.configuration.scripts_configuration import ContourOrbitConfig

from dataclasses import dataclass
from numpy import pi
import numpy as np

figsize = 13, 7  # Global window size
dpi = 100  # Global dpi
facecolor = "white"  # Global figure background color


@dataclass()
class ProfileEnergyContourConfig:
    # Figure keywords
    figsize: tuple = figsize
    dpi: int = dpi
    layout: str = "constrained"
    facecolor: str = facecolor
    projection: str | None = None  # None = default
    # Default parameter values
    thetalim: tuple = (-pi, pi)
    psilim: tuple = (0, 1.2)  # times psi_wall
    levels: int = 30
    E_units: str = "keV"
    flux_units: str = "NUMagnetic_flux"
    canmon_units: str = "NUCanonical_momentum"
    potential: bool = True
    wall: bool = True
    cursor: bool = True  # Mild performance hit
    show: bool = True
    # Colorbar
    numticks: int = 10
    cbarlabelsize: int = 12


@dataclass
class QfactorProfileConfig:
    # Figure keywords
    figsize: tuple = (13, 5)
    dpi: int = dpi
    layout: str = "constrained"
    facecolor: str = facecolor
    titlesize: float = 20
    titlecolor: str = "blue"
    # Default parameter values
    span: tuple = (0, 1.03)
    show: bool = True
    # Plot options
    points: int = 600
    wall_color: str = "red"
    qwall_color: str = "black"
    qwall_style: str = "--"
    psip_wall_color: str = "black"
    psip_wall_style: str = "--"
    labelsize: float = 10
    ax_title_size: float = 20


@dataclass
class EfieldProfileConfig:
    # Figure keywords
    figsize: tuple = (13, 5)
    dpi: int = dpi
    layout: str = "constrained"
    facecolor: str = "white"
    titlesize: float = 20
    titlecolor: str = "blue"
    # Default parameter values
    span: tuple = (0, 1.1)
    show: bool = True
    # Plot options
    points: int = 400
    wall_color: str = "red"
    labelsize: float = 10
    ax_title_size: float = 20
    linewidths: float = 2
    wall_linewidth: float = 1.5
    # Units options
    field_units: str = "kiloVolt/meters"
    potential_units: str = "kiloVolt"


@dataclass
class MagneticProfileConfig:
    # Figure keywords
    figsize: tuple = (13, 7)
    dpi: int = dpi
    layout: str = "constrained"
    facecolor: str = facecolor
    titlesize: float = 20
    titlecolor: str = "blue"
    # Default parameter values
    span: tuple = (0, 1.1)
    units: str = "NU"
    plot_derivatives: bool = True
    coord: str = "r"  # psi / r
    show: bool = True
    # Contour options
    grid_density: int = 400
    levels: int = 20
    bcmap: str = "managua"
    icmap: str = "managua"
    gcmap: str = "managua"
    locator: str = ""
    log_base: float = 1.00001
    # 2d plot options
    current_color: str = "b"
    derivative_color: str = "r"
    linewidth: float = 2
    # Label options
    labelsize: float = 15
    ax_title_size: float = 20
    ax_title_pad: float = 25


@dataclass
class PsiPthetaConfig:
    # Figure keywords
    figsize: tuple = (6, 5)
    dpi: int = dpi
    layout: str = "constrained"
    facecolor: str = facecolor
    titlesize: float = 20
    titlecolor: str = "blue"
    # Default parameter values
    flux_units: str = "NUMagnetic_flux"
    Ptheta_units: str = "NUCanonical_momentum"
    show: bool = True
    # Plot options
    linewidth: float = 3
    points: int = 600
    wall_color: str = "red"
    wall_style: str = "--"
    labelsize: float = 15
    ax_title_size: float = 15


@dataclass()
class ParticleEvolutionConfig:
    # Figure keywords
    figsize: tuple = figsize
    dpi: int = dpi
    layout: str = "constrained"
    facecolor: str = facecolor
    titlesize: float = 20
    titlecolor: str = "blue"
    # Default parameter values
    which: str = "all"
    units: str = "SI"
    percentage: int = 100
    show: bool = True
    # Scatter kw
    s: float = 0.2
    color: str = "blue"
    marker: str = "o"
    labelsize: int = 10
    labelpad: float = 8


@dataclass()
class ParticlePoloidalDrift:
    # Figure keywords
    figsize: tuple = figsize
    dpi: int = dpi
    layout: str = "constrained"
    facecolor: str = facecolor
    # Default parameter values
    projection: str | None = None  # None = default
    thetalim: tuple = (-pi, pi)
    psilim: str | tuple = "auto"  # times psi_wall, or "auto"
    levels: int = 30
    E_units: str = "keV"
    flux_units: str = "Tesla * meter^2"
    potential: bool = True
    initial: bool = True
    wall: bool = True
    cursor: bool = True  # Mild performance hit
    show: bool = True
    # Colorbar
    numticks: int = 10
    cbarlabelsize: int = 12


@dataclass
class AutoYspan:
    zoomout: float = 0.75
    hardylim: float = 3  # times psi_wall


@dataclass
class MachineCoordsContoursConfig:
    # Figure keywords
    figsize_B: tuple = (17, 8)
    figsize_E_flux: tuple = (11, 8)
    figsize_I: tuple = (11, 8)
    figsize_g: tuple = (11, 8)
    figsize_fp: tuple = (11, 8)
    dpi: int = dpi
    layout: str = "constrained"
    facecolor: str = "white"
    # Plots kwywords
    which: str = "fp E b i g"
    # E, flux figure keywords
    E_flux_suptitle_fontsize: float = 15
    E_flux_suptitle_color: str = "black"
    flux_units: str = "NUmf"
    E_units: str = "keV"
    # B figure keywords
    B_suptitle_fontsize: float = 15
    B_suptitle_color: str = "black"
    B_units: str = "Tesla"
    # I figure keywords
    I_suptitle_fontsize: float = 15
    I_suptitle_color: str = "black"
    I_units: str = "NUpc"
    # g figure keywords
    g_suptitle_fontsize: float = 15
    g_suptitle_color: str = "black"
    g_units: str = "NUpc"
    # Fixed Points figure keywords
    fp_suptitle_fontsize: float = 15
    fp_suptitle_color: str = "black"
    E_fp_units: str = "keV"
    # Numerical keywords
    parametric_density: int = 500
    # Contours keywords
    levels: int = 20


@dataclass
class FrequencyAnalysisPlotConfig(ContourOrbitConfig):
    scatter_figsize: tuple = figsize
    scatter_dpi: int = dpi
    scatter_size: float = 7
    add_hline: bool = True


@dataclass
class ParabolasPlotConfig:
    # Figure keywords
    figsize: tuple = figsize
    dpi: int = dpi
    layout: str = "constrained"
    facecolor: str = "white"
    linewidth: int = 2
    # Title keywords
    title_fontsize: float = 15
    title_color: str = "black"
    # Labels keywords
    xlabel_fontsize: float = 13
    xlabel_rotation: int = 0
    ylabel_fontsize: float = 13
    ylabel_rotation: int = 0
    # Legend keywords
    parabolas_legend: bool = True
    # Parabolas keywords
    enlim: tuple = (0, 3)
    Pzetalim: tuple = (-1, 1)  # result after division by psip_wall.m
    Pzeta_density: int = 1000
    TPB_density: int = 100
    plot_TPB: bool = False
    parabolas_color: str = "orange"
    TPB_X_color: str = "#E65100"
    TPB_O_color: str = "#1f77b4"
    TPB_X_linestyle: str = "solid"
    TPB_O_linestyle: str = "solid"
    TPB_X_markersize: float = 2
    TPB_O_markersize: float = 2
    # Dashed line keywords
    show_d_line: bool = True
    d_line_color: str = "black"
    d_linewidth: int = 1
    d_line_alplha: float = 0.5


@dataclass()
class BifurcationPlotConfig:
    # Figure keywords
    figsize: tuple = figsize
    dpi: int = dpi
    layout: str = "constrained"
    facecolor: str = "white"
    sharex: bool = True
    # x Label keywords
    xlabel_fontsize: float = 10
    # Suptitle keywords
    suptitle_fontsize: float = 15
    suptitle_color: str = "black"
    # Bifurcation keywords
    thetalim: tuple = (-np.pi, np.pi)
    psilim: tuple = (0, 1.8)
    plot_energy_bif: bool = True
    plot_ndfp: bool = False
    which_COM: str = "Pzeta"
    # Units
    energy_units: str = "NUJoule"
    flux_units: str = "NUmf"
    canmon_units: str = "NUcanmom"
