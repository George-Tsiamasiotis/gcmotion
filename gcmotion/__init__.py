from gcmotion.utils.logger_setup import logger
import matplotlib

# gtk3agg backend needs PyGObject, which needs a C compiler to be installed.
try:
    import gi
except ModuleNotFoundError:
    logger.warning("PyGobject not found. Using default backend")
    pass
else:
    logger.info("PyGObject availiable. using 'gtk3agg' backend")
    matplotlib.use("gtk3agg")

# Utilities
# Import the logger first
from gcmotion.utils.quantity_constructor import QuantityConstructor
from gcmotion.utils.get_size import get_size

# Tokamak Configuration Objects
from gcmotion.tokamak import qfactor
from gcmotion.tokamak import bfield
from gcmotion.tokamak import efield
from gcmotion.tokamak.reconstructed.initializers import (
    SmartPositiveInit,
    SmartNegativeInit,
    DivertorNegativeInit,
)

# Entities
from gcmotion.entities.tokamak import Tokamak
from gcmotion.entities.initial_conditions import InitialConditions
from gcmotion.entities.profile import Profile
from gcmotion.entities.particle import Particle

# Scripts
from gcmotion.scripts import events

# TEMP
from gcmotion.scripts.frequency_analysis.frequency_analysis import (
    frequency_analysis,
)


# main namespace
__all__ = [
    "logger_setup",
    "QuantityConstructor",
    "get_size",
    # Tokamak objects
    "SmartPositiveInit",
    "SmartNegativeInit",
    "DivertorNegativeInit",
    "qfactor",
    "bfield",
    "efield",
    # Entities
    "Tokamak",
    "InitialConditions",
    "Profile",
    "Particle",
    # Scripts
    "events",
    "frequency_analysis",
]
