import numpy as np
import pint
from gcmotion.utils.logger_setup import logger
from collections import deque
from gcmotion.entities.profile import Profile

type Quantity = pint.UnitRegistry.Quantity


def calc_parabolas(Pzetalim: Quantity, profile: Profile, Pzeta_density: int = 100):

    # Unpack parameters
    bfield = profile.bfield
    solverbNU = bfield.solverbNU

    # Turn Pzeta limits to [NU]
    PzetaminNU, PzetamaxNU = Pzetalim.to("NUcanonical_momentum").m
    PzetasNU = np.linspace(PzetaminNU, PzetamaxNU, Pzeta_density)
