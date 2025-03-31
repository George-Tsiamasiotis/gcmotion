import numpy as np
from pathlib import Path
from dataclasses import dataclass


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
    atol: float = 1e-10  # Scipy's default is 1e-6
    rtol: float = 1e-10  # Scipy's default is 1e-3


@dataclass
class ParabolasConfig:
    Pzetalim: tuple = (-1.5, 1)
    Pzeta_density: int = 1000
    TPB_density: int = 100


@dataclass
class NPeriodSolverConfig:
    event_variable: str = "theta"
    max_step = 40
    min_step_num = 50
    atol: float = 1e-12  # Scipy's default is 1e-6
    rtol: float = 1e-12  # Scipy's default is 1e-3
    stop_point_atol: float = 1e-3
    stop_point_rtol: float = 1e-3
    final_y_rtol: float = 1e-4
    max_recursion_depth: int = 40
    # NOTE: Time in NU to time out the solver, in case the initial conditions
    # are very close to a separatrix. This number is empirically derived and is
    # only checked during the first period.
    tstopNU: float = 3000


@dataclass
class NumericalDatasetsConfig:
    # Above 10-20 orbits seem to not conserve energy
    boozer_theta_downsampling_factor: int = 6
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


# -------------- Fixed Points - Bifurcation Config ---------------------------


@dataclass
class BifurcationPbarConfig(ProgressBarStyle):
    tqdm_enable: bool = True
    tqdm_desc: str = "Iterating through"  # + COM
    tqdm_unit: str = f"{'COM values':^12}"


@dataclass()
class FixedPointsConfig:
    thetalim: tuple = (-np.pi, np.pi)
    psilim: tuple = (0, 1.8)
    fp_method: str = "fsolve"
    dist_tol: float = 1e-3
    fp_ic_scan_tol: float = 5 * 1e-8
    ic_fp_theta_grid_density: int = 500
    ic_fp_psi_grid_density: int = 101
    fp_ic_scaling_factor: float = 90
    fp_random_init_cond: bool = False
    fp_info: bool = False
    fp_ic_info: bool = False
    fp_only_confined: bool = False


@dataclass()
class BifurcationConfig:
    bif_info: bool = False
    calc_energies: bool = False
    energy_units: str = "NUJoule"
    flux_units: str = "NUmf"
    energies_info: bool = False
    which_COM: str = "Pzeta"


# --------------- Reasonances Range (Omega Max) Configurations-----------------


@dataclass
class ResRangeConfig:
    freq_units_theta: str = "NUw0"
    freq_units_zeta: str = "NUw0"
    hessian_dtheta: float = 1e-5
    hessian_dpsi: float = 1e-5
    which_COM: str = "Pzeta"
