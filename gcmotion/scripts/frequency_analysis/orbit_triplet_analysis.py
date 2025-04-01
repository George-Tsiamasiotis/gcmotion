import pint
from collections import deque
from .contour_orbit import Orbit
from .lines_processing import (
    generate_contour_initial_conditions,
    generate_orbits,
)
from .contour_triplet_analysis import finalize_orbits

from gcmotion.entities.profile import Profile
from gcmotion.entities.particle import Particle
from gcmotion.entities.initial_conditions import InitialConditions
from gcmotion.configuration.scripts_configuration import (
    FrequencyAnalysisConfig,
)

type Quantity = pint.registry.Quantity


def orbit_triplet_analysis(
    main_contour: dict,
    profile: Profile,
    config: FrequencyAnalysisConfig,
):
    initial_conditions = generate_contour_initial_conditions(
        main_contour=main_contour,
        level=profile.ENU.magnitude,
    )

    particles = deque()
    for init in initial_conditions:
        theta0, _psi0 = init
        psi0 = profile.Q(_psi0, "NUMagnetic_flux")

        particle = create_particle(profile, theta0, psi0)
        particle.run(method="NPeriods", stop_after=1)
        particles.append(particle)

    orbits = deque()
    for p in particles:
        orbit = generate_orbits(p)
        orbits.append(orbit)

    for orb in orbits:
        orb.calculate_bbox()
        orb.classify_as_tp()
        orb.classify_as_cocu(profile)
        orb.str_dump()
        orb.pick_color()

    return orbits


def create_particle(profile: Profile, theta0: float, psi0: Quantity):

    tokamak = profile.tokamak
    init = InitialConditions(
        species=profile.species,
        muB=profile.muNU,
        Pzeta0=profile.PzetaNU,
        theta0=theta0,
        psi0=psi0,
        zeta0=0,  # Doesn't matter
    )
    return Particle(tokamak=tokamak, init=init)
