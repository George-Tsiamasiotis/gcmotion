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


# ============================ Frequency Analysis ============================


@dataclass
class FrequencyAnalysisConfig:
    tqdm_enable: bool = True
    tqdm_ascii: str = "-#"
    tqdm_colour: str = "green"
    tqdm_dynamic_ncols: bool = False
    tqdm_mu_desc: str = f"{'Iterating through mus':^28}"
    tqdm_pzeta_desc: str = f"{'Iterating through pzetas':^28}"
    tqdm_energy_desc: str = f"{'Iterating through energies':^28}"
    tqdm_mu_unit: str = f"{'mus':^10}"
    tqdm_pzeta_unit: str = f"{'Pzetas':^10}"
    tqdm_energy_unit: str = f"{'Energies':^10}"


@dataclass
class CalculateQkinConfig:
    pzeta_rtol: float = 1e-2


@dataclass
class CalculateOmegaThetaConfig:
    energy_rtol: float = 1e-3


@dataclass
class ContourGeneratorConfig:
    main_grid_density: int = 400
    local_grid_density: int = 100
    theta_expansion: float = 2
    psi_expansion: float = 2


@dataclass
class ContourOrbitConfig:
    inbounds_atol: float = 1e-7  # Must not be 0
    inbounds_rtol: float = 1e-7
    trapped_color: str = "red"
    copassing_color: str = "xkcd:light purple"
    cupassing_color: str = "xkcd:navy blue"
    undefined_color: str = "xkcd:blue"


# @dataclass
# class ContourFreqConfig:
#     # Arguements
#     plot_main_paths: bool = True
#     # Figures
#     figsize: tuple = (13, 7)
#     dpi = 100
#     layout: str = "constrained"
#     # Contour
#     levels: int = 200
#     log_base: float = 1.0000001
#     grid_density: int = 200
#     potential: bool = True
#     # Segments
#     is_inbounds_atol: float = 1e-7  # Must not be 0 when comparing with 0
#     trapped_color: str = "red"
#     copassing_color: str = "xkcd:light purple"
#     cupassing_color: str = "xkcd:navy blue"
#     undefined_color: str = "xkcd:blue"
#     scatter_size: float = 4
#     fullshow: bool = False
#     energy_rtol: float = 1e-5
#     pzeta_rtol: float = 1e-3
#     rho_sample_size: int = 10
#     omega_zeta_sample_size: int = 40
#     check_omega_attr: bool = True
#     # Misc
#     pbar: bool = True
#     tqdm_style: str = "-#"
#     tqdm_color: str = "green"
