from dataclasses import dataclass
from collections import deque
import numpy as np


@dataclass()
class FixedPointsConfig:
    fp_thetalim: tuple = (-np.pi, np.pi)
    fp_psilim: tuple = (0, 1.8)
    fp_method: str = "fsolve"
    dist_tol: float = 1e-3
    fp_ic_scan_tol: float = 5 * 1e-8
    ic_fp_theta_grid_density: int = 500
    ic_fp_psi_grid_density: int = 101
    fp_ic_scaling_factor: float = 70
    fp_random_init_cond: bool = False
    fp_info: bool = False
    fp_ic_info: bool = False
    fp_LAR_thetas: bool = False
    fp_only_confined: bool = False


@dataclass()
class FixedPointsPlotConfig:
    fp_plot_init_cond: bool = False
    flux_units: str = "Tesla * meter^2"


@dataclass()
class BifurcationConfig:
    bif_info: bool = False
    calc_energies: bool = False
    energy_units: str = "keV"
    energies_info: bool = False


@dataclass()
class BifurcationPlotConfig:
    thetalim: tuple = (-np.pi, np.pi)
    psilim: tuple = (0, 1.8)
    plot_energy_bif: bool = True
