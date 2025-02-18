import pint
import numpy as np
import matplotlib.pyplot as plt

from time import time
from tqdm import tqdm
from collections import deque
from contourpy import ContourGenerator
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator
from matplotlib.collections import LineCollection

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

    _plot_results(paths=contour_paths, config=config)

    # _plot_main_paths(contour_paths, config)
    # _plot_omega_thetas(contour_paths)
    # _plot_omega_zetas(contour_paths)
    # _plot_qkinetics(contour_paths)


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
        desc=f"{'Generating ContourOrbitss':^25}",
        unit=" paths",
        ascii=config.tqdm_style,
        colour=config.tqdm_color,
        disable=not config.pbar,
    ):
        lines = C.lines(level=energy)
        for points_array in lines:
            contour_paths.append(
                ContourOrbit(
                    E=energy,
                    ylim=metadata["ylim"],
                    vertices=points_array,
                )
            )

    dur = Q(time() - start, "seconds")
    logger.info(f"\tTook {dur:.4g~#P}.")

    return contour_paths


def _discard_invalid_paths(
    contour_paths: list[ContourOrbit],
) -> list[ContourOrbit]:
    r"""Validates and discards generated ContourOrbitss"""

    logger.info("\tValidating ContourOrbitss...")

    for path in contour_paths:
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

    upper_paths = deque()
    for points_array in upper_lines:
        upper_paths.append(
            ContourOrbit(
                E=None,
                ylim=ylim,
                vertices=points_array,
            )
        )

    lower_paths = deque()
    for points_array in lower_lines:
        lower_paths.append(
            ContourOrbit(
                E=None,
                ylim=ylim,
                vertices=points_array,
            )
        )

    upper_paths = _discard_invalid_paths(upper_paths)
    lower_paths = _discard_invalid_paths(lower_paths)
    _prepare_paths(upper_paths, profile)
    _prepare_paths(lower_paths, profile)

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
    PzetaUpperMetadata = Contours["PzetaUpperMetadata"]
    PzetaLowerMetadata = Contours["PzetaLowerMetadata"]

    upper_lines = CLower.lines(level=E)
    lower_lines = CUpper.lines(level=E)

    upper_paths = deque()
    for points_array in upper_lines:
        upper_paths.append(
            ContourOrbit(
                E=None,
                ylim=PzetaUpperMetadata["ylim"],
                vertices=points_array,
            )
        )

    lower_paths = deque()
    for points_array in lower_lines:
        lower_paths.append(
            ContourOrbit(
                E=None,
                ylim=PzetaLowerMetadata["ylim"],
                vertices=points_array,
            )
        )

    upper_paths = _discard_invalid_paths(upper_paths)
    lower_paths = _discard_invalid_paths(lower_paths)
    _prepare_paths(upper_paths, Contours["PzetaUpperProfile"])
    _prepare_paths(lower_paths, Contours["PzetaLowerProfile"])

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
    plt.plot(*main_path.vertices.T, color="r")
    plt.plot(*upper_path.vertices.T, color="b")
    plt.plot(*lower_path.vertices.T, color="b")
    plt.show()

    upper_path.calculate_Jtheta()
    lower_path.calculate_Jtheta()

    dPzeta = PzetaUpper - PzetaLower
    dJtheta = upper_path.Jtheta - lower_path.Jtheta

    main_path.qkinetic = dJtheta / dPzeta


def _calculate_omega_zeta(path: list[ContourOrbit]) -> None:

    try:
        path.omega_zeta = path.qkinetic * path.omega_theta
    except AttributeError:
        return


# ================================= Plots ===================================

# Manual lengend entries patches
trapped = Patch(color=global_config.trapped_color, label="Trapped")
copassing = Patch(color=global_config.copassing_color, label="Co-passing")
cupassing = Patch(color=global_config.cupassing_color, label="Counter-Passing")


def _plot_main_paths(lines: list[ContourOrbit], config):
    r"""Plots all segments."""

    if not config.plot_main_paths:
        return

    fig = plt.figure(**global_fig_kw)
    ax = fig.add_subplot()

    # See LineCollection example
    # Unpacking and creating a LineCollection is MUCH faster.
    seg_vertices = tuple((np.column_stack(line.vertices.T) for line in lines))
    colors = tuple(line.color for line in lines)

    collection = LineCollection(seg_vertices, colors=colors)
    ax.add_collection(collection)
    ax.set_xlim([-np.pi, np.pi])
    ax.set_ylim(lines[1].ylim)
    ax.set_title(r"Contour Segments")
    ax.set_xlabel(r"$\theta [radians]$")
    ax.set_ylabel(r"$P_\theta [NU]$")

    ax.legend(handles=[trapped, copassing, cupassing], loc="upper right")
    plt.show()


def _plot_omega_thetas(lines: list[ContourOrbit]):
    r"""Plots all omegas."""

    fig = plt.figure(**global_fig_kw)
    ax = fig.add_subplot()
    fig.suptitle(r"$\omega_\theta- Energy$")

    # See LineCollection example
    # Unpacking and creating a LineCollection is MUCH faster.
    omegas = tuple(line.omega_theta for line in lines)
    energies = tuple(line.E for line in lines)
    colors = tuple(line.color for line in lines)
    ax.scatter(energies, omegas, c=colors, s=global_config.scatter_size)

    ax.axhline(y=0, ls="--", lw=0.5, c="k")
    ax.set_xlabel("Energy [NU]")
    ax.set_ylabel(r"$\omega_\theta [\omega_0]$")

    ax.legend(handles=[trapped, copassing, cupassing])

    plt.show()


def _plot_omega_zetas(lines: list[ContourOrbit]):
    r"""Plots all omegas."""

    fig = plt.figure(**global_fig_kw)
    ax = fig.add_subplot()
    fig.suptitle(r"$\omega_\theta- Energy$")

    # See LineCollection example
    # Unpacking and creating a LineCollection is MUCH faster.
    omegas = tuple(line.omega_zeta for line in lines)
    energies = tuple(line.E for line in lines)
    colors = tuple(line.color for line in lines)
    ax.scatter(energies, omegas, c=colors, s=global_config.scatter_size)

    ax.axhline(y=0, ls="--", lw=0.5, c="k")
    ax.set_xlabel("Energy [NU]")
    ax.set_ylabel(r"$\omega_\theta [\omega_0]$")

    ax.legend(handles=[trapped, copassing, cupassing])

    plt.show()


def _plot_qkinetics(lines: list[ContourOrbit]):
    r"""Plots all omegas."""

    fig = plt.figure(**global_fig_kw)
    ax = fig.add_subplot()
    fig.suptitle(r"$\omega_\theta- Energy$")

    # See LineCollection example
    # Unpacking and creating a LineCollection is MUCH faster.
    qs = tuple(line.qkinetic for line in lines)
    energies = tuple(line.E for line in lines)
    colors = tuple(line.color for line in lines)
    ax.scatter(energies, qs, c=colors, s=global_config.scatter_size)

    ax.axhline(y=0, ls="--", lw=0.5, c="k")
    ax.set_xlabel("Energy [NU]")
    ax.set_ylabel(r"$\omega_\theta [\omega_0]$")

    ax.legend(handles=[trapped, copassing, cupassing])

    plt.show()
