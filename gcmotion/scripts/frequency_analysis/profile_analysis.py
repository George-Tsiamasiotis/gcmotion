r"""
Analyses a single *slice* of a Profile and tries to find all valid segments to
ccalculate their frequencies and qkinetics.

A slice is a contour graph over the ω-Ρθ space, with fixed μ and Ρζ.
Valid orbits consist of isoenergy lines that are fully contained inside the
graph, without getting cuttoff by its edges.

Functions
---------
"""

import numpy as np

from collections import deque

from gcmotion.entities.profile import Profile
import gcmotion.scripts.frequency_analysis.lines_processing as lp
from gcmotion.scripts.frequency_analysis.contour_orbit import ContourOrbit
from gcmotion.configuration.scripts_configuration import ProfileAnalysisConfig
from .plots import debug_plot_valid_orbits
from gcmotion.scripts.frequency_analysis.contour_generators import (
    local_contour,
)


def profile_analysis(
    main_contour: dict, profile: Profile, psilim=tuple, **kwargs
) -> list[ContourOrbit]:
    r"""Generates orbits that successfully calculated their frequency and
    finalizes them for analysis.

    Parameters
    ----------
    profile: Profile
        The profile to analyse. Both μ and Pζ must be defined.
    psilim: 2-tuple
        The area on which to search for orbits, relative to ψ_wall.
    """

    # Unpack Parameters
    config = ProfileAnalysisConfig()
    for key, value in kwargs.items():
        setattr(config, key, value)

    valid_orbits = lp.generate_valid_contour_orbits(
        main_contour=main_contour,
        level=profile.ENU.magnitude,
        config=config,
    )

    # For all the valid orbits, attempt to calculate their frequencies. Failed
    # attempts return None
    attempted_orbits = deque()
    for orbit in valid_orbits:
        attempted_orbits.append(
            calculate_frequencies(
                main_orbit=orbit,
                profile=profile,
                main_contour=main_contour,
                config=config,
            )
        )

    # Filter out failed attempts
    calculated_orbits = [orb for orb in attempted_orbits if orb is not None]

    finalized_orbits = finalize_orbits(
        calculated_orbits=calculated_orbits,
        main_profile=profile,
        config=config,
    )

    return finalized_orbits


def calculate_frequencies(
    main_orbit: ContourOrbit,
    profile: Profile,
    main_contour: dict,
    config: ProfileAnalysisConfig,
):
    r"""Calculates omega_theta, qkinetic and finally omega_zeta for a given
    orbit.

    Returns None when one of the calculations fails, and the orbit should be
    discarded.
    """

    # Step 3: Calculating frequencies for every contour orbit.
    main_orbit.mu = profile.muNU.m
    main_orbit.Pzeta = profile.PzetaNU.m

    # Omega_theta seems to be the fastest of the two, so try this one first
    # and abort if no omega is found.
    main_orbit.omega_theta = calculate_orbit_omegatheta(
        orbit=main_orbit,
        main_contour=main_contour,
        profile=profile,
        config=config,
    )
    if main_orbit.omega_theta is None:
        return None

    main_orbit.qkinetic = calculate_orbit_qkin(
        main_orbit=main_orbit, profile=profile, config=config
    )
    if main_orbit.qkinetic is None:
        return None

    main_orbit.omega_zeta = calculate_omegazeta(orbit=main_orbit)

    return main_orbit


def calculate_orbit_omegatheta(
    orbit: ContourOrbit,
    main_contour: dict,
    profile: Profile,
    config: ProfileAnalysisConfig,
) -> float:
    r"""Calculates omega_theta by evaluating the derivative dE/dJθ localy upon
    the orbit.

    Returns
    -------
    float or None
        The calculated omega_theta, if valid adjacent contours are found.
    """

    E = orbit.E
    Eupper = E * (1 + config.energy_rtol)
    Elower = E * (1 - config.energy_rtol)

    # Generate lower and upper orbits
    lower_orbits = lp.generate_valid_contour_orbits(
        main_contour=main_contour, level=Elower, config=config
    )
    upper_orbits = lp.generate_valid_contour_orbits(
        main_contour=main_contour, level=Eupper, config=config
    )

    # Return if orbit is about to disappear
    if len(lower_orbits) == 0 or len(upper_orbits) == 0:
        orbit.edge_orbit = True
        return
    else:
        orbit.edge_orbit = False

    lower_distances = [
        orbit.distance_from(lower_orbit.bbox) for lower_orbit in lower_orbits
    ]
    upper_distances = [
        orbit.distance_from(upper_orbit.bbox) for upper_orbit in upper_orbits
    ]

    lower_orbit = lower_orbits[np.argmin(lower_distances)]
    upper_orbit = upper_orbits[np.argmin(upper_distances)]

    # Calculate omega_theta
    for adjecent_orbit in [lower_orbit, upper_orbit]:
        adjecent_orbit.classify_as_tp()
        adjecent_orbit.close_segment()
        adjecent_orbit.convert_to_ptheta(profile.findPtheta, profile.Q)
        adjecent_orbit.calculate_Jtheta()

    dE = Eupper - Elower

    dJtheta = upper_orbit.Jtheta - lower_orbit.Jtheta

    omega_theta = dE / dJtheta
    return omega_theta


def calculate_orbit_qkin(
    main_orbit: ContourOrbit, profile: Profile, config: ProfileAnalysisConfig
) -> float:
    r"""Step 2: For each contour orbit found, calculate qkinetic.

    Returns
    -------
    float or None
        The calculated qkinetic, if valid adjacent contours are found.
    """

    # Calculate orbits bounding box and t/p classification
    main_orbit.calculate_bbox()
    main_orbit.classify_as_tp()

    # Create 2 local contours, 1 from each adjacent profile.
    Pzeta = profile.PzetaNU
    PzetaLower = Pzeta * (1 - config.pzeta_rtol)
    PzetaUpper = Pzeta * (1 + config.pzeta_rtol)

    profile.Pzeta = PzetaLower
    LowerContour = local_contour(profile=profile, orbit=main_orbit)
    profile.Pzeta = PzetaUpper
    UpperContour = local_contour(profile=profile, orbit=main_orbit)

    profile.Pzeta = Pzeta

    # Try to find contour lines with the same E in each Contour
    lower_orbits = lp.generate_valid_contour_orbits(
        main_contour=LowerContour, level=profile.ENU.m, config=config
    )
    upper_orbits = lp.generate_valid_contour_orbits(
        main_contour=UpperContour, level=profile.ENU.m, config=config
    )

    # Return if orbit is about to disappear
    if len(lower_orbits) == 0 or len(upper_orbits) == 0:
        main_orbit.edge_orbit = True
        return
    else:
        main_orbit.edge_orbit = False

    # Validate orbits, and calculate
    for orbit in lower_orbits:
        orbit.validate(LowerContour["psilim"])
        orbit.calculate_bbox()
    for orbit in upper_orbits:
        orbit.validate(UpperContour["psilim"])
        orbit.calculate_bbox()

    # Pick closest adjacent orbits
    lower_distances = [
        orbit.distance_from(lower_orbit.bbox) for lower_orbit in lower_orbits
    ]
    upper_distances = [
        orbit.distance_from(upper_orbit.bbox) for upper_orbit in upper_orbits
    ]

    lower_orbit = lower_orbits[np.argmin(lower_distances)]
    upper_orbit = upper_orbits[np.argmin(upper_distances)]

    # Classsify the 2 adjacent orbits as trapped/passing and and base points if
    # needed, convert psis to pthetas and calculate Jtheta.
    for orbit in (lower_orbit, upper_orbit):
        orbit.classify_as_tp()
        orbit.close_segment()
        orbit.convert_to_ptheta(profile.findPtheta, profile.Q)
        orbit.calculate_Jtheta()

    # Calculate qkinetic
    dJtheta = upper_orbit.Jtheta - lower_orbit.Jtheta
    dPzeta = PzetaUpper.m - PzetaLower.m

    # print(f"{upper_distances=}, {lower_distances=}")
    # print(f"{upper_orbit.xmin=}, {upper_orbit.xmax=}")
    # debug_plot_valid_orbits(profile, (lower_orbit, main_orbit, upper_orbit))
    # print(f"{main_orbit.E=}")

    qkinetic = -dJtheta / dPzeta
    if abs(qkinetic) < 4:
        return qkinetic


def calculate_omegazeta(orbit: ContourOrbit):

    omegazeta = orbit.qkinetic * orbit.omega_theta
    return omegazeta


def finalize_orbits(
    calculated_orbits: list[ContourOrbit],
    main_profile: Profile,
    config: ProfileAnalysisConfig,
) -> list[ContourOrbit]:

    finalized_orbits = deque()
    for orbit in calculated_orbits:
        if config.cocu_classification and orbit.passing:
            orbit.classify_as_cocu(profile=main_profile)
        orbit.pick_color()
        orbit.str_dump()
        finalized_orbits.append(orbit)

    return finalized_orbits
