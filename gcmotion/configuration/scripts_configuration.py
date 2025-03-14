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
class SolverConfig:
    atol: float = 1e-9  # Scipy's default is 1e-6
    rtol: float = 1e-8  # Scipy's default is 1e-3


@dataclass
class ParabolasConfig:
    Pzetalim: tuple = (-1.5, 1)
    Pzeta_density: int = 1000
    TPB_density: int = 100


@dataclass
class NumericalDatasetsConfig:
    # Above 10-20 orbits seem to not conserve energy
    boozer_theta_downsampling_factor: int = 1
    currents_spline_order: int = 3
    qfactor_spline_order: int = 3


@dataclass
class PrecomputedConfig:
    psi_max: int = 2  # Max spline extend relative to psi_wall
    hyp2f1_density: int = 1000


# -------------- Fixed Points - Bifurcation Config ---------------------------


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
    freq_units: str = "NUw0"
    hessian_dtheta: float = 1e-5
    hessian_dpsi: float = 1e-5
    which_COM: str = "Pzeta"
