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
    boozer_theta_downsampling_factor: int = 10


@dataclass
class PrecomputedConfig:
    psi_max: int = 2  # Max spline extend relative to psi_wall
    hyp2f1_density: int = 1000
