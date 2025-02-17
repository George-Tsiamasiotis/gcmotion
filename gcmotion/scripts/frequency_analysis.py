import pint
import contourpy
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
from gcmotion.scripts.utils.contour_lines import ContourPath

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
    C, energy_span, ylim = _create_contour(profile, psilim, config=config)

    # Generate all the lines upon which we will calculate the frequencies
    contour_paths = _generate_contour_paths(C, energy_span, ylim, config)

    # Discard out of bounds / cutoff contours
    contour_paths = _discard_invalid_paths(contour_paths)

    # Classify, add base points and pick color
    _prepare_paths(paths=contour_paths, profile=profile)

    # Calculate ωθ, ωζ and qkinetic
    _calculate_frequencies(contour_paths, C, profile, config)

    _plot_main_paths(contour_paths, config)

    _plot_omega_thetas(contour_paths)


def _create_contour(
    profile: Profile, psilim: tuple, config
) -> tuple[ContourGenerator, tuple, tuple]:
    r"""Creates a ContourGenerator from contourpy, as well as a couple of
    needed quantities."""

    logger.info("\tCreating Contour Generator...")
    start = time()

    thetalim = profile.Q((-tau, tau), "radians")
    psilim = profile.Q(psilim, "psi_wall").to("NUMagnetic_flux")
    psi_grid, theta_grid = np.meshgrid(
        np.linspace(psilim[0], psilim[1], config.grid_density),
        np.linspace(thetalim[0], thetalim[1], config.grid_density),
    )

    energy_grid = profile.findEnergy(
        psi=psi_grid,
        theta=theta_grid.m,
        units="NUJoule",
        potential=config.potential,
    )
    ptheta_grid = profile.findPtheta(
        psi=psi_grid,
        units="NUCanonical_momentum",
    )

    C = contourpy.contour_generator(
        x=theta_grid.m,
        y=ptheta_grid.m,
        z=energy_grid.m,
        line_type="Separate",
        fill_type="OuterCode",
    )
    energy_span = (energy_grid.min().m, energy_grid.max().m)
    ylim = (ptheta_grid.m[0][0], ptheta_grid.m[0][-1])

    dur = Q(time() - start, "seconds")
    logger.info(f"\tTook {dur:.4g~#P}.")

    return C, energy_span, ylim


# ======================= ContourPath Instantiation =========================


def _generate_contour_paths(
    C: ContourGenerator, energy_span: tuple, ylim: tuple, config
) -> list[ContourPath]:
    r"""Extracts paths from the Contour Generator on different energy levels
    and creates ContourPath objects."""

    logger.info("\tGenerating Contour Paths...")
    start = time()

    locator = LogLocator(base=config.log_base, numticks=config.levels)
    locator.MAXTICKS = 40000
    energy_levels = locator.tick_values(*energy_span)

    contour_paths = deque()

    for energy in tqdm(
        iterable=energy_levels,
        desc=f"{'Generating ContourPaths':^25}",
        unit=" paths",
        ascii=config.tqdm_style,
        colour=config.tqdm_color,
        disable=not config.pbar,
    ):
        lines = C.lines(level=energy)
        for points_array in lines:
            contour_paths.append(
                ContourPath(
                    E=energy,
                    ylim=ylim,
                    vertices=points_array,
                )
            )

    dur = Q(time() - start, "seconds")
    logger.info(f"\tTook {dur:.4g~#P}.")

    return contour_paths


def _discard_invalid_paths(
    contour_paths: list[ContourPath],
) -> list[ContourPath]:
    r"""Validates and discards generated ContourPaths"""

    logger.info("\tValidating ContourPaths...")

    for path in contour_paths:
        path.validate()

    return [path for path in contour_paths if path.valid]


def _prepare_paths(paths: list[ContourPath], profile: Profile) -> None:
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
    contour_paths: list, C: ContourGenerator, profile: Profile, config
) -> None:
    r"""Iterates over all main ContourPaths and calcluates ωθ, ωζ and qkin."""

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
        _calculate_omega_theta(line, C, profile, config)

    logger.enable("gcmotion")

    dur = Q(time() - start, "seconds")
    logger.info(f"\tTook {dur:.4g~#P}.")


def _calculate_omega_theta(
    main_path: ContourPath, C: ContourGenerator, profile: Profile, config
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
            ContourPath(
                E=None,
                ylim=ylim,
                vertices=points_array,
            )
        )

    lower_paths = deque()
    for points_array in lower_lines:
        lower_paths.append(
            ContourPath(
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


# ================================= Plots ===================================

# Manual lengend entries patches
trapped = Patch(color=global_config.trapped_color, label="Trapped")
copassing = Patch(color=global_config.copassing_color, label="Co-passing")
cupassing = Patch(color=global_config.cupassing_color, label="Counter-Passing")


def _plot_main_paths(lines: list[ContourPath], config):
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


def _plot_omega_thetas(lines: list[ContourPath]):
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
