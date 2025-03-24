r"""
Analyses a single *slice* of a Profile and tries to find all valid segments to
calculate their frequencies and qkinetics.

A slice is a contour graph over the ω-Ρθ space, with fixed μ and Ρζ.

Functions
---------
profile_triplet_analysis():
    Top level function: Generates the main θ-Ρθ-E contour for a given triplet
    and returns all the valid found orbits, with all frequencies calculated.
calculate_frequencies():
    For a single orbit: Calculates its bounding box, classifies it as
    trapped-passing, and calls the functions that calculate the frequencies and
    qkinetic. Depending on the configuration, it can skip the calculations of
    the frequencies, or skip all trapped or passing orbits.
finalize_orbits():
    Calculates final labels for every orbit that succesfully calculated its
    frequencies.

Module-level variables
----------------------

single_contour_orbits, double_contour_orbits:
    Used to track the number of calls to each contouring method.


"""

from collections import deque
from typing import Callable
from gcmotion.entities.profile import Profile

from .contour_orbit import ContourOrbit
from .lines_processing import generate_valid_contour_orbits
from .frequency_calculations import (
    calculate_orbit_omegatheta_single,
    calculate_orbit_omegatheta_double,
    calculate_orbit_qkinetic_single,
    calculate_orbit_qkinetic_double,
    calculate_omegazeta,
)


from gcmotion.configuration.scripts_configuration import (
    FrequencyAnalysisConfig,
)

# Module-level variables used for tracking the calls of each method. Their
# values are handled and reset in the FrequencyAnalysis Class.
single_contour_orbits = 0
double_contour_orbits = 0


def profile_triplet_analysis(
    main_contour: dict,
    profile: Profile,
    psilim: tuple,
    config: FrequencyAnalysisConfig,
) -> list[ContourOrbit]:
    r"""Top level function: Generates the main θ-Ρθ-E contour for a given
    triplet and returns all the valid found orbits.

    Returns a list of 0 up to multiple ContourOrbit objects, depending on the
    valid contours it found.
    """

    valid_orbits = generate_valid_contour_orbits(
        main_contour=main_contour,
        level=profile.ENU.magnitude,
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
    config: FrequencyAnalysisConfig,
):
    r"""Calculates all frequencies for a given (validated) orbit.

    For a single orbit: Cclassifies it as trapped-passing, and calls the
    functions that calculate the frequencies and qkinetic. Depending on the
    configuration, it can skip the calculations of the frequencies, or skip all
    trapped or passing orbits.

    Returns None when one of the calculations fails, and the orbit should be
    discarded.
    """
    global single_contour_orbits
    global double_contour_orbits

    # Calculating frequencies for every contour orbit.
    main_orbit.mu = profile.muNU.m
    main_orbit.Pzeta = profile.PzetaNU.m
    main_orbit.Jzeta = main_orbit.Pzeta

    # Calculate orbits bounding box and t/p classification
    # co/cu classification must be done before closing the segment
    main_orbit.classify_as_tp()
    if config.cocu_classification and main_orbit.passing:
        main_orbit.classify_as_cocu(profile=profile)

    if main_orbit.trapped and config.skip_trapped:
        return None
    if main_orbit.passing and config.skip_passing:
        return None
    if main_orbit.copassing and config.skip_copassing:
        return None
    if main_orbit.cupassing and config.skip_cupassing:
        return None

    # Decide Method
    if (
        main_orbit.vertices.shape[0] < config.max_vertices_method_switch
        or abs(main_orbit.Pzeta) < config.max_pzeta_method_switch
    ):
        omega_theta_method: Callable = calculate_orbit_omegatheta_double
        qkinetic_method: Callable = calculate_orbit_qkinetic_double
        double_contour_orbits += 1
    else:
        omega_theta_method: Callable = calculate_orbit_omegatheta_single
        qkinetic_method: Callable = calculate_orbit_qkinetic_single
        single_contour_orbits += 1

    # Omega_theta seems to be the fastest of the two, so try this one first
    # and abort if no omega is found.
    if config.calculate_omega_theta:
        main_orbit.omega_theta = omega_theta_method(
            main_orbit=main_orbit,
            main_contour=main_contour,
            profile=profile,
            config=config,
        )
        if main_orbit.omega_theta is None:
            return None

    if config.calculate_qkinetic:
        main_orbit.qkinetic = qkinetic_method(
            main_orbit=main_orbit, profile=profile, config=config
        )
        if main_orbit.qkinetic is None:
            return None

    if config.calculate_omega_theta and config.calculate_qkinetic:
        main_orbit.omega_zeta = calculate_omegazeta(orbit=main_orbit)

    return main_orbit


def finalize_orbits(
    calculated_orbits: list[ContourOrbit],
    main_profile: Profile,
    config: FrequencyAnalysisConfig,
) -> list[ContourOrbit]:
    r"""Calculates label attributes for each orbit."""

    finalized_orbits = deque()
    for orbit in calculated_orbits:
        orbit.pick_color()
        orbit.str_dump()
        finalized_orbits.append(orbit)

    return finalized_orbits
