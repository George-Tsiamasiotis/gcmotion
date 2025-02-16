from pathlib import Path
from dataclasses import dataclass


@dataclass
class LoggerConfig:
    sink: str | Path = "./log.log"
    level: str = "TRACE"
    mode: str = "w"  # (w)rite / (a)ppend
    format: str = "timedelta"  # timedelta / default
    colorize: bool = True
    backtrace: bool = True
    # format prefixes
    module_prefix: bool = False
    file_prefix: bool = False
    name_prefix: bool = False


@dataclass
class SolverConfig:
    atol: float = 1e-9  # Scipy's default is 1e-6
    rtol: float = 1e-8  # Scipy's default is 1e-3


@dataclass
class ContourFreqConfig:
    # Arguements
    show_master_contour: bool = False
    show_close_segments: bool = False
    show_segments: bool = True
    # Figures
    figsize: tuple = (13, 7)
    dpi = 100
    layout: str = "constrained"
    # Contour
    levels: int = 10  # For demo
    locator: str = "log"
    log_base: float = 1.0000001
    grid_density: int = 200
    show_base_points: bool = False
    potential: bool = True
    # Segments
    is_inbounds_atol: float = 1e-7  # Must not be 0 when comparing with 0
    trapped_color: str = "red"
    copassing_color: str = "xkcd:light purple"
    cupassing_color: str = "xkcd:navy blue"
    undefined_color: str = "key"
    scatter_size: float = 4
    fullshow: bool = False
    energy_rtol: float = 1e-5
    rho_sample_size: int = 10
    omega_zeta_sample_size: int = 40
    check_omega_attr: bool = True
    # Misc
    pbar: bool = True
    tqdm_style: str = "-#"
    tqdm_color: str = "green"
