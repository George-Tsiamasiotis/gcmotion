from collections import deque

from gcmotion.scripts.frequency_analysis.contour_orbit import (
    ContourOrbit,
)


def generate_valid_contour_orbits(main_contour: dict, level: float, config):
    r"""[Steps 1-1e] Creates the contour lines from contourpy's
    ContourGenerator for a *specific* level. Then creates a ContourObrit object
    out of every segment, calculates their bounding boxes, validates them, and
    returns the valid ones.
    """

    # Step 1a: Generate lines and return if none found
    isoenergy_lines = main_contour["C"].lines(level=level)

    if len(isoenergy_lines) == 0:
        return []

    # Step 1b: Generate ContourOrbits from lines
    isoenergy_orbits = _generate_contour_orbits(isoenergy_lines, level)

    # Step 1c: Calculate bounding boxes and validate
    for orbit in isoenergy_orbits:
        orbit.calculate_bbox()
        orbit.validate(psilim=main_contour["psilim"])

    # Step 1e: Discrad invalid orbits
    valid_orbits = [orbit for orbit in isoenergy_orbits if orbit.valid]

    return valid_orbits


def _generate_contour_orbits(
    isoenergy_lines: list,
    level: float,
) -> list[ContourOrbit]:
    r"""Extracts isoenergy lines from the Contour Generator and creates
    ContourOrbits objects.
    """

    # list of ContourOrbits
    isoenergy_orbits = deque()

    for vertices in isoenergy_lines:
        isoenergy_orbits.append(ContourOrbit(vertices=vertices, E=level))

    return isoenergy_orbits
