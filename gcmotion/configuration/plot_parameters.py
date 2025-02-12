from dataclasses import dataclass
from numpy import pi

figsize = 13, 7  # Global window size
dpi = 100  # Global dpi


@dataclass()
class ProfileEnergyContourConfig:
    # Figure keywords
    figsize: tuple = figsize
    dpi: int = dpi
    layout: str = "constrained"
    facecolor: str = "lightskyblue"
    projection: str | None = None  # None = default
    # Default parameter values
    thetalim: tuple = (-pi, pi)
    psilim: tuple = (0, 1.2)  # times psi_wall
    levels: int = 30
    E_units: str = "keV"
    flux_units: str = "Tesla * meter^2"
    canmon_units: str = "NUCanonical_momentum"
    potential: bool = True
    wall: bool = True
    cursor: bool = True  # Mild performance hit
    show: bool = True
    # Colorbar
    numticks: int = 10
    cbarlabelsize: int = 12


@dataclass()
class ProfilePzetaContourConfig:
    # Figure keywords
    figsize: tuple = figsize
    dpi: int = dpi
    layout: str = "constrained"
    facecolor: str = "lightskyblue"
    # Default parameter values
    zetalim: tuple = (-pi, pi)
    psilim: tuple = (0, 1.2)  # times psi_wall
    levels: int = 30
    Pzeta_units: str = "Tesla * meter^2"
    flux_units: str = "Tesla * meter^2"
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
    facecolor: str = "lightskyblue"
    titlesize: float = 20
    titlecolor: str = "blue"
    # Default parameter values
    span: tuple = (0, 1.1)
    show: bool = True
    # Plot options
    points: int = 200
    wall_color: str = "red"
    qwall_color: str = "black"
    qwall_style: str = "--"
    psip_wall_color: str = "black"
    psip_wall_style: str = "--"
    labelsize: float = 10
    ax_title_size: float = 20


@dataclass
class MagneticProfileConfig:
    # Figure keywords
    figsize: tuple = (13, 7)
    dpi: int = dpi
    layout: str = "constrained"
    facecolor: str = "lightskyblue"
    titlesize: float = 20
    titlecolor: str = "blue"
    # Default parameter values
    span: tuple = (0, 1.1)
    units: str = "NU"
    plot_derivatives: bool = True
    coord: str = "r"  # psi / r
    show: bool = True
    # Contour options
    grid_density: int = 200
    levels: int = 20
    bcmap: str = "inferno"
    icmap: str = "viridis"
    gcmap: str = "viridis"
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


@dataclass()
class ParticleEvolutionConfig:
    # Figure keywords
    figsize: tuple = figsize
    dpi: int = dpi
    layout: str = "constrained"
    facecolor: str = "lightskyblue"
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
    facecolor: str = "lightskyblue"
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
