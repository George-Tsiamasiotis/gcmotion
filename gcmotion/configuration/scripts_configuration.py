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
class NumericalDatasetsConfig:
    # Above 10-20 orbits seem to not conserve energy
    boozer_theta_downsampling_factor: int = 5
    currents_spline_order: int = 3
    qfactor_spline_order: int = 3


@dataclass
class PrecomputedConfig:
    psi_max: int = 2  # Max spline extend relative to psi_wall
    hyp2f1_density: int = 1000


# ============================ Frequency Analysis ============================


@dataclass
class FrequencyAnalysisConfig:
    tqdm_enable: bool = True
    tqdm_ascii: str = "-#"
    tqdm_colour: str = "green"
    tqdm_dynamic_ncols: bool = False
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
class ProfileAnalysisConfig:
    pzeta_rtol: float = 1e-3
    energy_rtol: float = 1e-3
    cocu_classification: bool = True


@dataclass
class ContourGeneratorConfig:
    main_grid_density: int = 400
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
