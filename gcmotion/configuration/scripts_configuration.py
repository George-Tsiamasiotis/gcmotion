from pathlib import Path
from dataclasses import dataclass
import numpy as np


@dataclass
class LoggerConfig:
    sink: str | Path = "./log.log"
    level: str = "TRACE"
    mode: str = "w"  # (w)rite / (a)ppend
    format: str = "timedelta"  # timedelta / default
    colorize: bool = False
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
    tqdm_matrix_desc: str = f"{'Iterating through triplets':^28}"
    tqdm_matrix_unit: str = f"{'triplets':^8}"


@dataclass
class FrequencyAnalysisConfig:
    print_tokamak: bool = True
    qkinetic_cutoff: float = 10
    pzeta_min_step: float = 2e-4
    passing_pzeta_rstep: float = 1e-3  # 1e-3 seems to work best
    trapped_pzeta_rstep: float = 1e-3  # 1e-3 seems to work best
    energy_rstep: float = 1e-3  # 1e-3 seems to work best
    cocu_classification: bool = True
    calculate_omega_theta: bool = True
    calculate_qkinetic: bool = True
    skip_trapped: bool = False
    skip_passing: bool = False
    skip_copassing: bool = False
    skip_cupassing: bool = False
    # Minimum number of main orbit vertices, to switch to double contour method
    max_vertices_method_switch: int = 40
    # Maximum abs(Pzeta value, below which to switch to double contour method
    max_pzeta_method_switch: float = 0.000
    # dynamic minimum energy
    relative_upper_E_factor: float = 1.1
    logspace_len: int = 50
    trapped_min_num: int = 1
    # Contour Generation
    main_grid_density: int = 1000  # Diminishing results after 1800
    local_grid_density: int = 100
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
