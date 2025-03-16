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
class ProgressBarStyle:
    tqdm_ascii: str = "-#"
    tqdm_colour: str = "green"
    tqdm_dynamic_ncols: bool = False
    tqdm_smoothing: float = 0.15


@dataclass
class SolverConfig:
    atol: float = 1e-9  # Scipy's default is 1e-6
    rtol: float = 1e-8  # Scipy's default is 1e-3


@dataclass
class NumericalDatasetsConfig:
    # Above 10-20 orbits seem to not conserve energy
    boozer_theta_downsampling_factor: int = 10
    currents_spline_order: int = 3
    qfactor_spline_order: int = 3


@dataclass
class PrecomputedConfig:
    psi_max: int = 2  # Max spline extend relative to psi_wall
    hyp2f1_density: int = 1000


# ============================ Frequency Analysis ============================


@dataclass
class FrequencyAnalysisPbarConfig(ProgressBarStyle):
    tqdm_enable: bool = True
    # Cartesian Mode
    tqdm_mu_desc: str = f"{'Iterating through mus':^28}"
    tqdm_pzeta_desc: str = f"{'Iterating through pzetas':^28}"
    tqdm_energy_desc: str = f"{'Iterating through energies':^28}"
    tqdm_mu_unit: str = f"{'mus':^10}"
    tqdm_pzeta_unit: str = f"{'Pzetas':^10}"
    tqdm_energy_unit: str = f"{'Energies':^10}"
    # Matrix Mode
    tqdm_mu_Pzeta_desc: str = f"{'Iterating through mus/Pzetas':^28}"


@dataclass
class FrequencyAnalysisConfig:
    qkinetic_cutoff: float = 10
    pzeta_rtol: float = 1e-3  # 1e-3 seems to work best
    energy_rtol: float = 1e-3  # 1e-3 seems to work best
    cocu_classification: bool = True
    calculate_omega_theta: bool = True
    calculate_qkinetic: bool = True
    skip_trapped: bool = False
    skip_passing: bool = False
    # dynamic minimum energy
    relative_upper_E_factor: float = 1.1
    logspace_len: int = 50
    trapped_min_num: int = 1


@dataclass
class ContourGeneratorConfig:
    main_grid_density: int = 1200  # Diminishing results after 1800
    local_grid_density: int = 100
    centered_grid_density: int = 100
    theta_expansion: float = 1.2
    psi_expansion: float = 1.2


@dataclass
class ContourOrbitConfig:
    inbounds_atol: float = 1e-7  # Must not be 0
    inbounds_rtol: float = 1e-7
    trapped_color: str = "red"
    copassing_color: str = "xkcd:light purple"
    cupassing_color: str = "xkcd:navy blue"
    undefined_color: str = "xkcd:blue"
