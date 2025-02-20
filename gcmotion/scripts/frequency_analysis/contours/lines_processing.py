from collections import deque

from gcmotion.entities.profile import Profile
from gcmotion.scripts.frequency_analysis.contours.contour_orbits import (
    ContourOrbit,
)


def generate_contour_orbits(Contour: dict, level: float, config):
    r"""[Steps 1-1e]"""

    # Step 1a: Generate lines and return if none found
    isoenergy_lines = Contour["C"].lines(level=level)
    print(f"Isoenergy lines found: {len(isoenergy_lines)}")

    if len(isoenergy_lines) == 0:
        return None

    # Step 1b: Generate ContourOrbits from lines
    isoenergy_orbits = _generate_contour_orbits(isoenergy_lines, level)

    # Step 1c: Calculate bounding boxes
    for orbit in isoenergy_orbits:
        orbit.calculate_bbox()

    # Step 1d: Validate orbits
    for orbit in isoenergy_orbits:
        orbit.validate(ylim=Contour["ylim"])

    # Step 1e: Discrad invalid orbits
    valid_isoenergy_orbits = [
        orbit for orbit in isoenergy_orbits if orbit.valid
    ]
    print(f"Valid Isoenergy Orbits found: {len(valid_isoenergy_orbits)}")

    return valid_isoenergy_orbits


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


def discard_invalid_orbits(
    contour_orbits: list[ContourOrbit],
) -> list[ContourOrbit]:
    r"""Validates and discards generated ContourOrbits"""

    for orbit in contour_orbits:
        orbit._bbox_extends()
        orbit.validate()

    return [orbit for orbit in contour_orbits if orbit.valid]


def prepare_orbits(orbits: list[ContourOrbit], profile: Profile) -> None:
    r"""Classifies, adds base points in passing particles and picks the orbit
    color.

    Passing 'profile=None' skips co-/counter-passing classification, which is
    not needed when calculating Upper and Lower Orbits.
    """

    for orbit in orbits:
        orbit.classify(profile=profile)
        orbit.close_segment()
        if profile is None:
            continue
        orbit.pick_color()
