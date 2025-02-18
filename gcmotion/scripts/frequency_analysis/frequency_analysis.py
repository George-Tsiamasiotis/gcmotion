import pint
import numpy as np

from time import time
from tqdm import tqdm
from collections import deque
from contourpy import ContourGenerator
from matplotlib.path import Path
from matplotlib.ticker import LogLocator

from gcmotion.utils.logger_setup import logger
from gcmotion.entities.profile import Profile
from gcmotion.configuration.scripts_configuration import ContourFreqConfig
from gcmotion.scripts.frequency_analysis.contour_orbits import ContourOrbit
from gcmotion.scripts.frequency_analysis.plots import _plot_results
from gcmotion.scripts.frequency_analysis.contour_generators import (
    _create_contours,
)


Q = pint.get_application_registry().Quantity
tau = 2 * np.pi
global_config = ContourFreqConfig()
global_fig_kw = {
    "figsize": global_config.figsize,
    "dpi": global_config.dpi,
    "layout": global_config.layout,
}


def frequency_analysis(profile: Profile, psilim=tuple, **kwargs) -> None:

    logger.info("==> Beginning Frequency Analysis...")

    # Unpack default params and replace with passed kwargs if given.
    config = ContourFreqConfig()
    for key, value in kwargs.items():
        setattr(config, key, value)

    # Create Contour Generator
    Contours: dict = _create_contours(profile, psilim, config=config)

    # Generate all the lines upon which we will calculate the frequencies
    contour_paths = _generate_contour_paths(Contours, config)

    # Discard out of bounds / cutoff contours
    contour_paths = _discard_invalid_paths(contour_paths)

    # Classify, add base points and pick color
    _prepare_paths(paths=contour_paths, profile=profile)

    # Calculate ωθ, ωζ and qkinetic
    _calculate_frequencies(contour_paths, Contours, profile, config)

    # Discard faulty orbits
    contour_paths = [
        path
        for path in contour_paths
        if (
            hasattr(path, "omega_theta")
            and hasattr(path, "qkinetic")
            and hasattr(path, "omega_zeta")
        )
    ]

    undef_num = len(
        [path for path in contour_paths if getattr(path, "undefined", False)]
    )
    msg = f"Undefined passing found = {undef_num}"
    print(msg)
    logger.warning(msg)

    _plot_results(paths=contour_paths, config=config)


# ======================= ContourOrbits Instantiation =========================


def _generate_contour_paths(
    Contours: dict, config: ContourFreqConfig
) -> list[ContourOrbit]:
    r"""Extracts paths from the Contour Generator on different energy levels
    and creates ContourOrbits objects."""

    logger.info("\tGenerating Contour Paths...")
    start = time()

    C = Contours["EnergyContour"]
    metadata = Contours["EnergyMetadata"]

    locator = LogLocator(base=config.log_base, numticks=config.levels)
    locator.MAXTICKS = 40000
    energy_levels = locator.tick_values(*metadata["energy_span"])

    contour_paths = deque()

    for energy in tqdm(
        iterable=energy_levels,
        desc=f"{'Generating ContourOrbits':^25}",
        unit=" paths",
        ascii=config.tqdm_style,
        colour=config.tqdm_color,
        disable=not config.pbar,
    ):
        lines = C.lines(level=energy)
        contour_paths += _unpack_cg_lines(
            lines=lines, ylim=metadata["ylim"], E=energy
        )

    dur = Q(time() - start, "seconds")
    logger.info(f"\tTook {dur:.4g~#P}.")

    return contour_paths


def _unpack_cg_lines(
    lines: list, ylim: tuple, E: float = None
) -> list[ContourOrbit]:
    r"""Creates ContourObrit objects from the output of the contour generator's
    cg.lines(level=level).

    This only works if the generator's line_type is 'SeperateCode'.
    """
    contour_paths = deque()

    for point_array in lines:
        contour_paths.append(
            ContourOrbit(E=E, ylim=ylim, vertices=point_array)
        )

    return contour_paths


def _discard_invalid_paths(
    contour_paths: list[ContourOrbit],
) -> list[ContourOrbit]:
    r"""Validates and discards generated ContourOrbitss"""

    logger.info("\tValidating ContourOrbitss...")

    for path in contour_paths:
        path._bbox_extends()
        path.validate()

    return [path for path in contour_paths if path.valid]


def _prepare_paths(paths: list[ContourOrbit], profile: Profile) -> None:
    r"""Classifies, adds base points in passing particles and picks the orbit
    color."""

    logger.info("\tClassifying Paths...")
    start = time()

    for path in paths:
        path.classify(profile=profile)
        path.close_segment()
        if profile is None:
            continue
        path.pick_color()

    dur = Q(time() - start, "seconds")
    logger.info(f"\tTook {dur:.4g~#P}.")


# ====================== Segment Frequency Calculation ========================


def _calculate_frequencies(
    contour_paths: list, Contours: dict, profile: Profile, config
) -> None:
    r"""Iterates over all main ContourOrbitss and calcluates ωθ, ωζ and qkin."""

    logger.info("\tCalculating Frequencies...")
    start = time()

    logger.disable("gcmotion")
    for line in tqdm(
        iterable=contour_paths,
        desc=f"{'Calculating frequencies':^25}",
        unit=" freq",
        ascii=config.tqdm_style,
        colour=config.tqdm_color,
        disable=not config.pbar,
    ):
        _calculate_omega_theta(
            line, Contours["EnergyContour"], profile, config
        )
        _calculate_qkinetic(line, Contours, config)
        _calculate_omega_zeta(line)

    logger.enable("gcmotion")

    dur = Q(time() - start, "seconds")
    logger.info(f"\tTook {dur:.4g~#P}.")


def _calculate_omega_theta(
    main_path: ContourOrbit, C: ContourGenerator, profile: Profile, config
) -> None:

    ylim = main_path.ylim
    E = main_path.E
    Eupper = E * (1 + config.energy_rtol)
    Elower = E * (1 - config.energy_rtol)

    upper_lines = C.lines(level=Eupper)
    lower_lines = C.lines(level=Elower)

    upper_paths = _unpack_cg_lines(lines=upper_lines, ylim=ylim, E=E)
    lower_paths = _unpack_cg_lines(lines=lower_lines, ylim=ylim, E=E)

    upper_paths = _discard_invalid_paths(upper_paths)
    lower_paths = _discard_invalid_paths(lower_paths)
    # We dont need co/counterpassing here
    _prepare_paths(upper_paths, profile=None)
    _prepare_paths(lower_paths, profile=None)

    upper_distances = tuple(
        main_path.distance_from((path.xmin, path.ymin)) for path in upper_paths
    )
    lower_distances = tuple(
        main_path.distance_from((path.xmin, path.ymin)) for path in lower_paths
    )

    try:
        upper_path = upper_paths[np.argmin(upper_distances)]
        lower_path = lower_paths[np.argmin(lower_distances)]
    except ValueError:
        return

    dE = Eupper - Elower

    upper_path.calculate_Jtheta()
    lower_path.calculate_Jtheta()
    dJtheta = upper_path.Jtheta - lower_path.Jtheta

    main_path.omega_theta = dE / dJtheta


def _calculate_qkinetic(
    main_path: ContourOrbit,
    Contours: dict,
    config: ContourFreqConfig,
) -> None:

    E = main_path.E
    PzetaUpper = Contours["PzetaUpper"]
    PzetaLower = Contours["PzetaLower"]
    CUpper = Contours["PzetaUpperContour"]
    CLower = Contours["PzetaLowerContour"]
    ylim_upper = Contours["PzetaUpperMetadata"]["ylim"]
    ylim_lower = Contours["PzetaLowerMetadata"]["ylim"]

    # Contour Generator lists of vertices
    upper_lines = CLower.lines(level=E)
    lower_lines = CUpper.lines(level=E)

    # Now lists of ContourObrits. Note that we must pass the new grid's ylim in
    # order to classify them correctly
    upper_paths = _unpack_cg_lines(lines=upper_lines, ylim=ylim_upper, E=E)
    lower_paths = _unpack_cg_lines(lines=lower_lines, ylim=ylim_lower, E=E)

    upper_paths = _discard_invalid_paths(upper_paths)
    lower_paths = _discard_invalid_paths(lower_paths)
    # We dont need co/counterpassing here
    _prepare_paths(upper_paths, None)
    _prepare_paths(lower_paths, None)

    upper_distances = tuple(
        main_path.distance_from((path.xmin, path.ymin)) for path in upper_paths
    )
    lower_distances = tuple(
        main_path.distance_from((path.xmin, path.ymin)) for path in lower_paths
    )

    try:
        upper_path = upper_paths[np.argmin(upper_distances)]
        lower_path = lower_paths[np.argmin(lower_distances)]
    except ValueError:
        return

    # Show Pzeta Contours
    # plt.plot(*main_path.vertices.T, color="r")
    # plt.plot(*upper_path.vertices.T, color="b")
    # plt.plot(*lower_path.vertices.T, color="b")
    # plt.show()
    #
    upper_path.calculate_Jtheta()
    lower_path.calculate_Jtheta()

    dPzeta = PzetaUpper - PzetaLower
    dJtheta = upper_path.Jtheta - lower_path.Jtheta

    qkinetic = -dJtheta / dPzeta
    if abs(qkinetic) > 20:
        return
    main_path.qkinetic = qkinetic


def _calculate_omega_zeta(path: list[ContourOrbit]) -> None:

    try:
        path.omega_zeta = path.qkinetic * path.omega_theta
    except AttributeError:
        return
