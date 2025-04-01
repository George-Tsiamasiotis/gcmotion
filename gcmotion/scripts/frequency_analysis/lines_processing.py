r"""
================
Lines Processing
================

1. Extract contour lines from ContourGenerator.
2. Create a ContourOrbit out of every line.
3. Calculate ContourOrbit's bounding box.
4. Validate the ContourOrbit.
5. Return a list with all valid ContourObrits.

"""

import numpy as np
from collections import deque
from gcmotion.entities.particle import Particle
from .contour_orbit import Orbit, ContourOrbit


def generate_contour_initial_conditions(
    main_contour: dict,
    level: float,
):

    # Generate lines and return if none found
    isoenergy_lines = main_contour["C"].lines(level=level)

    # Grab first point of every line
    initial_conditions = [line[0] for line in isoenergy_lines]

    return initial_conditions


def generate_orbits(particle: Particle):

    from gcmotion.plot.particle_evolution import particle_evolution

    orbit = Orbit(
        E=particle.ENU.m,
        vertices=np.array((particle.theta.m, particle.psiNU.m)).T,
    )
    # particle_evolution(particle, which="theta psi rho")

    orbit.rho = particle.rhoNU.m
    orbit.omega_theta = particle.omega_thetaNU.m
    orbit.omega_zeta = particle.omega_zetaNU.m
    orbit.qkinetic = particle.qkinetic

    return orbit


def generate_valid_contour_orbits(
    main_contour: dict,
    level: float,
) -> list[ContourOrbit]:
    r"""Creates the contour lines from contourpy's ContourGenerator for a
    *specific* level. Then creates a ContourObrit object out of every segment,
    calculates their bounding boxes, validates them, and returns the valid
    ones.
    """

    # Generate lines and return if none found
    isoenergy_lines = main_contour["C"].lines(level=level)

    if len(isoenergy_lines) == 0:
        return []

    # Generate ContourOrbits from lines
    isoenergy_orbits = generate_contour_orbits(isoenergy_lines, level)

    # Calculate bounding boxes and validate
    for orbit in isoenergy_orbits:
        orbit.calculate_bbox()
        orbit.validate(psilim=main_contour["psilim"])

    # Discard invalid orbits
    valid_orbits = [orbit for orbit in isoenergy_orbits if orbit.valid]

    return valid_orbits


def generate_contour_orbits(
    isoenergy_lines: list[ContourOrbit],
    level: float,
) -> list[ContourOrbit]:
    r"""Creates a ContourOrbit out of every extracted contour line."""

    isoenergy_orbits = deque()

    for vertices in isoenergy_lines:
        isoenergy_orbits.append(ContourOrbit(vertices=vertices, E=level))

    return isoenergy_orbits
