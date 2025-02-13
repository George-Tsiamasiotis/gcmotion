from dataclasses import dataclass
import numpy as np

figsize = 13, 7  # Global window size
dpi = 100  # Global dpi


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
class BifurcationConfig:
    bif_info: bool = False
    calc_energies: bool = False
    energy_units: str = "keV"
    energies_info: bool = False
    which_COM: str = "Pzeta"
