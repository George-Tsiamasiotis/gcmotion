r"""
======================
Frequency Calculations
======================

Methods for calculating ωθ and qkinetic, for single-contour and double-contour
modes.
"""

import numpy as np
from .contour_orbit import ContourOrbit
from gcmotion.entities.profile import Profile
from .contour_generators import local_contour
from .lines_processing import generate_valid_contour_orbits
from gcmotion.configuration.scripts_configuration import (
    FrequencyAnalysisConfig,
)

# from gcmotion.utils.logger_setup import logger


def calculate_orbit_omegatheta_single(
    main_orbit: ContourOrbit,
    main_contour: dict,
    profile: Profile,
    config: FrequencyAnalysisConfig,
) -> float | None:
    r"""Calculates omega_theta by evaluating the derivative dE/dJθ localy upon
    the orbit, **using only the upper adjacent orbit**.

    Returns
    -------
    float or None
        The calculated omega_theta, if valid adjacent contours are found, else
        None.

    """

    E = main_orbit.E
    # Multiply by 2 to use the same tolerance with double contour method
    Eupper = E * (1 + 2 * config.energy_rstep)

    # Generate upper orbits
    upper_orbits = generate_valid_contour_orbits(
        main_contour=main_contour, level=Eupper
    )

    upper_distances = [
        main_orbit.distance_from(upper_orbit.bbox)
        for upper_orbit in upper_orbits
    ]

    try:
        upper_orbit = upper_orbits[np.argmin(upper_distances)]
    except ValueError:
        return None

    # Calculate omega_theta
    for orbit in [main_orbit, upper_orbit]:
        orbit.classify_as_tp()
        orbit.close_segment()
        orbit.convert_to_ptheta(profile.findPtheta, profile.Q)
        orbit.calculate_Jtheta()

    # Local derivative
    dE = Eupper - E
    dJtheta = upper_orbit.Jtheta - main_orbit.Jtheta
    omega_theta = dE / dJtheta

    return omega_theta


def calculate_orbit_omegatheta_double(
    main_orbit: ContourOrbit,
    main_contour: dict,
    profile: Profile,
    config: FrequencyAnalysisConfig,
) -> float:
    r"""Calculates omega_theta by evaluating the derivative dE/dJθ localy upon
    the orbit, **using both lower and upper adjacent orbits**.

    Returns
    -------
    float or None
        The calculated omega_theta, if valid adjacent contours are found, else
        None.
    """

    E = main_orbit.E
    Eupper = E * (1 + config.energy_rstep)
    Elower = E * (1 - config.energy_rstep)

    # Generate upper orbits only
    upper_orbits = generate_valid_contour_orbits(
        main_contour=main_contour, level=Eupper
    )
    lower_orbits = generate_valid_contour_orbits(
        main_contour=main_contour, level=Elower
    )

    upper_distances = [
        main_orbit.distance_from(upper_orbit.bbox)
        for upper_orbit in upper_orbits
    ]
    lower_distances = [
        main_orbit.distance_from(lower_orbit.bbox)
        for lower_orbit in lower_orbits
    ]

    try:
        upper_orbit = upper_orbits[np.argmin(upper_distances)]
        lower_orbit = lower_orbits[np.argmin(lower_distances)]
    except ValueError:
        return None

    # Calculate omega_theta
    for orb in [lower_orbit, upper_orbit]:
        orb.classify_as_tp()
        orb.close_segment()
        orb.convert_to_ptheta(profile.findPtheta, profile.Q)
        orb.calculate_Jtheta()

    # Local Derivative
    dE = Eupper - Elower
    dJtheta = upper_orbit.Jtheta - lower_orbit.Jtheta
    omega_theta = dE / dJtheta

    return omega_theta


def calculate_orbit_qkinetic_single(
    main_orbit: ContourOrbit, profile: Profile, config: FrequencyAnalysisConfig
) -> float:
    r"""Calculates qkinetic by evaluating the derivative dJθ/dPζ localy upon
    the orbit, **using only the upper adjacent orbit**.

    Returns
    -------
    float or None
        The calculated qkinetic, if valid adjacent contours are found.
    """

    if main_orbit.passing:
        rtol = config.passing_pzeta_rstep
    else:
        rtol = config.trapped_pzeta_rstep

    # Create 2 local contours, 1 from each adjacent profile.
    # Multiply by 2 to use the same tolerance with double contour method
    Pzeta = profile.PzetaNU
    atol = np.sign(Pzeta.m) * config.pzeta_min_step * Pzeta.units
    PzetaUpper = atol + Pzeta * (1 + rtol)
    # if Pzeta.m > -0.003:
    #     logger.trace(
    #         f"atol = {atol.m}, Pzeta = {Pzeta.m}, "
    #         f"PzetaUpper = {PzetaUpper.m}"
    #     )
    #     logger.trace(f"Diff = {PzetaUpper.m - Pzeta.m}")

    profile.Pzeta = PzetaUpper
    UpperContour = local_contour(
        profile=profile, orbit=main_orbit, config=config
    )

    profile.Pzeta = Pzeta

    # Try to find contour lines with the same E in each Contour
    upper_orbits = generate_valid_contour_orbits(
        main_contour=UpperContour, level=profile.ENU.m
    )

    # Validate orbits, and calculate
    for orbit in upper_orbits:
        orbit.validate(UpperContour["psilim"])
        orbit.calculate_bbox()

    # Pick closest adjacent orbits
    upper_distances = [
        main_orbit.distance_from(upper_orbit.bbox)
        for upper_orbit in upper_orbits
    ]

    try:
        upper_orbit = upper_orbits[np.argmin(upper_distances)]
    except ValueError:
        return None

    # Classsify the 2 adjacent orbits as trapped/passing and and base points if
    # needed, convert psis to pthetas and calculate Jtheta.
    for orbit in [main_orbit, upper_orbit]:
        orbit.classify_as_tp()
        orbit.close_segment()
        orbit.convert_to_ptheta(profile.findPtheta, profile.Q)
        orbit.calculate_Jtheta()

    # Calculate qkinetic
    dJtheta = upper_orbit.Jtheta - main_orbit.Jtheta
    dPzeta = PzetaUpper.m - Pzeta.m

    qkinetic = -dJtheta / dPzeta
    if abs(qkinetic) < config.qkinetic_cutoff:
        return qkinetic


def calculate_orbit_qkinetic_double(
    main_orbit: ContourOrbit, profile: Profile, config: FrequencyAnalysisConfig
) -> float:
    r"""Calculates qkinetic by evaluating the derivative dJθ/dPζ localy upon
    the orbit, **using both lower and upper adjacent orbits**.

    Returns
    -------
    float or None
        The calculated qkinetic, if valid adjacent contours are found.
    """

    if main_orbit.passing:
        rtol = config.passing_pzeta_rstep
    else:
        rtol = config.passing_pzeta_rstep

    # Create 2 local contours, 1 from each adjacent profile.
    Pzeta = profile.PzetaNU
    atol = np.sign(Pzeta.m) * config.pzeta_min_step * Pzeta.units
    PzetaLower = atol + Pzeta * (1 - rtol)
    PzetaUpper = atol + Pzeta * (1 + rtol)
    # if Pzeta.m > -0.003:
    #     logger.trace(
    #         f"atol = {atol.m}, Pzeta = {Pzeta.m}, "
    #         f"PzetaUpper = {PzetaUpper.m}, PzetaLower = {PzetaLower.m}"
    #     )
    #     logger.trace(f"Diff = {PzetaUpper.m - Pzeta.m}")

    profile.Pzeta = PzetaLower
    LowerContour = local_contour(
        profile=profile, orbit=main_orbit, config=config
    )
    profile.Pzeta = PzetaUpper
    UpperContour = local_contour(
        profile=profile, orbit=main_orbit, config=config
    )

    profile.Pzeta = Pzeta

    # Try to find contour lines with the same E in each Contour
    lower_orbits = generate_valid_contour_orbits(
        main_contour=LowerContour, level=profile.ENU.m
    )
    upper_orbits = generate_valid_contour_orbits(
        main_contour=UpperContour, level=profile.ENU.m
    )

    # Validate orbits, and calculate
    for orbit in lower_orbits:
        orbit.validate(LowerContour["psilim"])
        orbit.calculate_bbox()
    for orbit in upper_orbits:
        orbit.validate(UpperContour["psilim"])
        orbit.calculate_bbox()

    # Pick closest adjacent orbits
    lower_distances = [
        main_orbit.distance_from(lower_orbit.bbox)
        for lower_orbit in lower_orbits
    ]
    upper_distances = [
        main_orbit.distance_from(upper_orbit.bbox)
        for upper_orbit in upper_orbits
    ]

    try:
        lower_orbit = lower_orbits[np.argmin(lower_distances)]
        upper_orbit = upper_orbits[np.argmin(upper_distances)]
    except ValueError:
        return None

    # Classsify the 2 adjacent orbits as trapped/passing and and base points if
    # needed, convert psis to pthetas and calculate Jtheta.
    for orbit in [lower_orbit, upper_orbit]:
        orbit.classify_as_tp()
        orbit.close_segment()
        orbit.convert_to_ptheta(profile.findPtheta, profile.Q)
        orbit.calculate_Jtheta()

    # Calculate qkinetic
    dJtheta = upper_orbit.Jtheta - lower_orbit.Jtheta
    dPzeta = PzetaUpper.m - PzetaLower.m

    qkinetic = -dJtheta / dPzeta
    if abs(qkinetic) < config.qkinetic_cutoff:
        return qkinetic


def calculate_omegazeta(orbit: ContourOrbit):
    r"""Simple multiplication :)"""

    omegazeta = orbit.qkinetic * orbit.omega_theta
    return omegazeta
