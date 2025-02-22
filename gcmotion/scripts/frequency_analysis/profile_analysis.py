import numpy as np
from copy import deepcopy

from gcmotion.configuration.scripts_configuration import (
    CalculateQkinConfig,
    CalculateOmegaThetaConfig,
)
from gcmotion.scripts.frequency_analysis.contours.contour_orbit import (
    ContourOrbit,
)
from gcmotion.scripts.frequency_analysis.contours.contour_generators import (
    main_contour,
    local_contour,
)
from gcmotion.scripts.frequency_analysis.contours.lines_processing import (
    generate_contour_orbits,
)

from gcmotion.scripts.frequency_analysis.plots import debug_plot_valid_orbits
from gcmotion.entities.profile import Profile


def profile_analysis(profile: Profile, psilim: tuple = (0.05, 1), **kwargs):

    config = CalculateQkinConfig()
    for key, value in kwargs.items():
        setattr(config, key, value)

    # Step 1: Generate valid isoenergy orbits. Store them in Profile for
    # conviniency.
    MainContour = main_contour(profile=profile, psilim=psilim)

    valid_orbits = generate_contour_orbits(
        Contour=MainContour, level=profile.ENU.m, config=config
    )

    if len(valid_orbits) == 0:
        # print("No valid orbits found.")
        return None

    # debug_plot_valid_orbits(profile, valid_orbits, psilim)

    # Step 2: Create adjecent Pzeta profiles. Fastest way I could find.
    lower_profile, upper_profile = deepcopy(profile), deepcopy(profile)
    lower_profile.PzetaNU = profile.PzetaNU * (1 - config.pzeta_rtol)
    upper_profile.PzetaNU = profile.PzetaNU * (1 + config.pzeta_rtol)
    assert upper_profile is not lower_profile

    Profiles = {
        "Main": profile,
        "Lower": lower_profile,
        "Upper": upper_profile,
    }

    # Step 3: Calculating frequencies for every contour orbit.
    for orbit in valid_orbits:
        orbit.Pzeta = profile.PzetaNU.m
        orbit.mu = profile.muNU.m

        orbit.omega_theta = calculate_omegatheta(
            orbit=orbit,
            main_contour=MainContour,
            profile=Profiles["Main"],
        )
        if orbit.omega_theta is None:
            continue

        orbit.qkinetic = calculate_orbit_qkin(orbit=orbit, profiles=Profiles)
        if orbit.qkinetic is None:
            continue

        orbit.omega_zeta = calculate_omegazeta(orbit=orbit)
        del orbit.vertices

    return_orbits = [
        orbit for orbit in valid_orbits if hasattr(orbit, "omega_zeta")
    ]

    if len(return_orbits) > 0:
        return return_orbits
    else:
        return None


def calculate_orbit_qkin(orbit: ContourOrbit, profiles: dict) -> float:
    r"""Step 2: For each contour orbit found, calculate qkinetic."""

    config = CalculateQkinConfig()

    # Step 2a: Calculate orbits bounding box and t/p classification
    orbit.calculate_bbox()
    orbit.classify_as_tp()

    # Step 2b: Create 2 local contours, 1 from each adjacent profile.
    LowerContour = local_contour(profile=profiles["Lower"], orbit=orbit)
    UpperContour = local_contour(profile=profiles["Upper"], orbit=orbit)
    profiles["Lower"].psilim = LowerContour["psilim"]
    profiles["Upper"].psilim = UpperContour["psilim"]

    # Step 2c: Try to find contour lines with the same E in each Contour
    lower_orbits = generate_contour_orbits(
        Contour=LowerContour, level=profiles["Main"].ENU.m, config=config
    )
    upper_orbits = generate_contour_orbits(
        Contour=UpperContour, level=profiles["Main"].ENU.m, config=config
    )

    # print(f"Lower orbits found: {len(lower_orbits)}")
    # print(f"Upper orbits found: {len(upper_orbits)}")

    # debug_plot_valid_orbits(
    #     profiles["Lower"], (*lower_orbits, orbit, *upper_orbits)
    # )
    # debug_plot_valid_orbits(profiles["Lower"], lower_orbits)
    # debug_plot_valid_orbits(profiles["Upper"], upper_orbits)

    # Step 2d: Return if orbit is about to disappear
    if len(lower_orbits) == 0 or len(upper_orbits) == 0:
        orbit.edge_orbit = True
        return
    else:
        orbit.edge_orbit = False

    # Step 2e: Validate orbits, and calculate
    for orbit in lower_orbits:
        orbit.validate(LowerContour["psilim"])
        orbit.calculate_bbox()
    for orbit in upper_orbits:
        orbit.validate(UpperContour["psilim"])
        orbit.calculate_bbox()

    # Step 2f: pick closest adjacent orbits
    lower_distances = (
        orbit.distance_from(lower_orbit.bbox) for lower_orbit in lower_orbits
    )
    upper_distances = (
        orbit.distance_from(upper_orbit.bbox) for upper_orbit in upper_orbits
    )

    lower_orbit = lower_orbits[np.argmin(lower_distances)]
    upper_orbit = upper_orbits[np.argmin(upper_distances)]

    # Step 3a: Classsify the 2 adjacent orbits as trapped/passing and and base
    # points if needed
    # Step 3b: Converter psis to pthetas
    # Step 3c : Calculate Jtheta
    for orbit, profile in zip(
        [lower_orbit, upper_orbit], [profiles["Lower"], profiles["Upper"]]
    ):
        orbit.classify_as_tp()
        orbit.close_segment()
        orbit.convert_to_ptheta(profile)
        orbit.calculate_Jtheta()

    # Step 3d: Calculate qkinetic
    dJtheta = upper_orbit.Jtheta - lower_orbit.Jtheta
    dPzeta = profiles["Upper"].PzetaNU.m - profiles["Lower"].PzetaNU.m

    qkinetic = -dJtheta / dPzeta
    if abs(qkinetic) < 8:
        return qkinetic


def calculate_omegatheta(
    orbit: ContourOrbit, main_contour: dict, profile: Profile
):
    config = CalculateOmegaThetaConfig()

    psilim = main_contour["psilim"]

    E = orbit.E
    Eupper = E * (1 + config.energy_rtol)
    Elower = E * (1 - config.energy_rtol)

    lower_orbits = generate_contour_orbits(
        Contour=main_contour, level=Elower, config=config
    )
    upper_orbits = generate_contour_orbits(
        Contour=main_contour, level=Eupper, config=config
    )

    if len(lower_orbits) == 0 or len(upper_orbits) == 0:
        orbit.edge_orbit = True
        return
    else:
        orbit.edge_orbit = False

    for orbit in lower_orbits:
        orbit.validate(psilim)
        orbit.calculate_bbox()
    for orbit in upper_orbits:
        orbit.validate(psilim)
        orbit.calculate_bbox()

    lower_distances = (
        orbit.distance_from(lower_orbit.bbox) for lower_orbit in lower_orbits
    )
    upper_distances = (
        orbit.distance_from(upper_orbit.bbox) for upper_orbit in upper_orbits
    )

    lower_orbit = lower_orbits[np.argmin(lower_distances)]
    upper_orbit = upper_orbits[np.argmin(upper_distances)]

    for adjecent_orbit in [lower_orbit, upper_orbit]:
        adjecent_orbit.classify_as_tp()
        adjecent_orbit.close_segment()
        adjecent_orbit.convert_to_ptheta(profile)
        adjecent_orbit.calculate_Jtheta()

    dE = Eupper - Elower

    upper_orbit.calculate_Jtheta()
    upper_orbit.calculate_Jtheta()
    dJtheta = upper_orbit.Jtheta - lower_orbit.Jtheta

    omega_theta = dE / dJtheta
    return omega_theta


def calculate_omegazeta(orbit: ContourOrbit):

    omegazeta = orbit.qkinetic * orbit.omega_theta
    return omegazeta
