from dataclasses import dataclass
import numpy as np


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
    fp_info: bool = True
    fp_ic_info: bool = True
    fp_only_confined: bool = False


@dataclass()
class BifurcationConfig:
    bif_info: bool = True
    calc_energies: bool = False
    energy_units: str = "keV"
    energies_info: bool = False
    which_COM: str = "Pzeta"
