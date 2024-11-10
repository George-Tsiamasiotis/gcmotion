# fmt: off

# Import the logger first
from gcmotion.utils import logger_setup

from gcmotion.utils.setup_pint import setup_pint
from gcmotion.utils.get_size import get_size

from gcmotion.classes.particle import Particle
from gcmotion.classes.collection import Collection
from gcmotion.scripts import events

from gcmotion.tokamak import qfactor
from gcmotion.tokamak import bfield
from gcmotion.tokamak import efield

from gcmotion.plotters.time_evolution import time_evolution
from gcmotion.plotters.qfactor_profile import qfactor_profile
from gcmotion.plotters.magnetic_profile import magnetic_profile
from gcmotion.plotters.electric_profile import electric_profile
from gcmotion.plotters.tokamak_profile import tokamak_profile
from gcmotion.plotters.drift import drift
from gcmotion.plotters.drifts import drifts
from gcmotion.plotters.energy_contour import energy_contour
from gcmotion.plotters.parabolas import parabolas
from gcmotion.plotters.poloidal_cut import poloidal_cut
from gcmotion.plotters.torus2d import torus2d
from gcmotion.plotters.torus3d import torus3d

from gcmotion.plotters.collection_drift import collection_drift
from gcmotion.plotters.collection_energy_contour import collection_energy_contour
from gcmotion.plotters.collection_drifts import collection_drifts
from gcmotion.plotters.collection_poloidal_cut import collection_poloidal_cut
from gcmotion.plotters.collection_parabolas import collection_parabolas
#import gcmotion.scripts.animation as animation

__all__ = [
    "logger_setup",
    "setup_pint",
    "get_size",
    "Particle",
    "Collection",
    "qfactor",
    "bfield",
    "efield",
    "time_evolution",
    "qfactor_profile",
    "magnetic_profile",
    "electric_profile",
    "tokamak_profile",
    "drift",
    "drifts",
    "energy_contour",
    "parabolas",
    "poloidal_cut",
    "torus2d",
    "torus3d",
    "collection_drift",
    "collection_drifts",
    "collection_energy_contour",
    "collection_poloidal_cut",
    "collection_parabolas",
    "events",
    #"animation",
]
