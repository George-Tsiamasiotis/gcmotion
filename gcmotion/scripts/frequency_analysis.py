import pint
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from tqdm import tqdm
from time import time
from collections import deque
from matplotlib.patches import Patch
from matplotlib.contour import QuadContourSet
from matplotlib.collections import LineCollection

from gcmotion.utils.logger_setup import logger
from gcmotion.entities.profile import Profile
from gcmotion.configuration.scripts_configuration import ContourFreqConfig
from gcmotion.scripts.utils.contour_segment import ContourSegment
from gcmotion.scripts.utils._contours import _ptheta_energy_contour

ureg = pint.get_application_registry()
Q = ureg.Quantity

config = ContourFreqConfig()
global_fig_kw = {
    "figsize": config.figsize,
    "dpi": config.dpi,
    "layout": config.layout,
}


def frequency_analysis(profile: Profile, psilim: list, **kwargs):

    logger.info("==> Beginning Frequency Analysis...")
    # Unpack it here and update it, since only this function is to be called
    # externaly
    for key, value in kwargs.items():
        setattr(config, key, value)

    # Create the Contour object, which also contains the grid and the ylim
    # tuple
    C = _ptheta_energy_contour(profile=profile, psilim=psilim, config=config)

    # Unpack Contour paths into ContourSegments and store them in a list.
    segs = _unpack_and_return_paths(C=C, progress=config.pbar)

    # Validate ContourSegments and discard invalid ones.
    segs = _validate_and_return(segments=segs)

    # Prepare the segments to calculate their frequencies.
    _prepare_segments(segments=segs, profile=profile)

    _plot_segments(segs)

    _calculate_segment_frequencies(segs, C.data, profile)

    # WARN: this might be necessary. Sometimes, contours very close to the grid
    # limits cannot find a valid segment close to them, and fail to calculate
    # their omega.
    segs = [seg for seg in segs if hasattr(seg, "omega_theta")]

    _plot_omega_thetas(segments=segs)
    # _plot_omega_zetas(segments=segs)
    # _plot_qkinetic(segments=segs)

    logger.info("Frequency analysis complete.")

    return segs


def _unpack_and_return_paths(
    C: QuadContourSet, progress: bool = True
) -> list[ContourSegment]:
    r"""Iterates over C.get_paths(), unpacking each one and creating a
    ContourSegment, assigning to it the correct energy. It also discards all
    empty segments, which are unavoidably yielded from the gererator.

    Returns
    -------
    list
        A list containing *all* the extracted ContourSegments.

    """

    segments = deque()

    # Iterate over allsegs and create ContourSegments
    logger.info("==> Extracting ContourSegments")
    start = time()

    # HACK: C.allsegs does the exact same thing, though we can't surely
    # assign the correct energy to each seg. However, this method also
    # relies on the contour object return the Paths in rising energy
    # levels, since C.levels is sorted.

    for E, composite_path in tqdm(
        iterable=zip(C.levels, C.get_paths()),
        total=len(C.allsegs),
        desc=f"{"Creating ContourSegments":^25}",
        unit=" segs",
        ascii=config.tqdm_style,
        colour=config.tqdm_color,
        disable=not progress,
    ):
        segment_genenerator = composite_path._iter_connected_components()

        for segment in segment_genenerator:
            if segment.vertices.shape == (0, 2):  # Discard empty segments
                continue

            new_segment = ContourSegment(segment=segment, E=E, ylim=C.ylim)
            segments.append(new_segment)
    del C

    number = len(segments)
    duration = Q(time() - start, "seconds")
    logger.info(f"\tUnpacked {number} segments. Took {duration:.4g~#P}.")

    return segments


def _unpack_and_return_paths_optim(
    C: QuadContourSet, progress: bool = True
) -> list[ContourSegment]:
    r"""Same as _unpack_and_return_paths, but without the clutter. To be used
    in _calculate_omega_theta(). Actually is ~10% faster!"""

    for E, path in zip(C.levels, C.get_paths()):

        for segment in path._iter_connected_components():
            if segment.vertices.shape == (0, 2):  # Discard empty segments
                continue

            yield ContourSegment(segment=segment, E=E, ylim=C.ylim)


# ====================== ContourSegment Instantiation ========================


def _validate_and_return(
    segments: list[ContourSegment],
) -> tuple[ContourSegment]:
    r"""Makes every segment validate itself and returns the valid ones.

    validate() checks if the segment is fully inbounds and doesn't get cutoff.

    Returns
    -------
    tuple
        A list containing all the valid ContourSegments

    """
    logger.info("==> Validating Segments...")

    start = time()
    for seg in segments:
        seg.validate()

    valid_segments = tuple(seg for seg in segments if seg.valid)

    duration = Q(time() - start, "seconds")
    logger.info(
        f"\tKept {len(valid_segments)}/{len(segments)} valid segments. "
        f"Took {duration:.4g~#P}."
    )

    return valid_segments


def _prepare_segments(
    segments: list[ContourSegment],
    profile: Profile = None,
) -> list[ContourSegment]:
    r"""Makes all ContourSegments classify themselves"""

    logger.info("\tPreparing Segments...")

    start = time()
    for seg in segments:
        seg.classify(profile)
    classify_dur = Q(time() - start, "seconds")

    start = time()
    for seg in segments:
        seg.close_segment()
    close_segment_dur = Q(time() - start, "seconds")

    start = time()
    for seg in segments:
        seg.calculate_Jtheta()
    calculate_J_dur = Q(time() - start, "seconds")

    start = time()
    for seg in segments:
        seg.pick_color()
    color_picking_dur = Q(time() - start, "seconds")

    total = classify_dur + close_segment_dur + calculate_J_dur
    logger.info(
        f"\tSegments classified and areas calculated. Took {total:.4g~#P}."
    )
    logger.info(f"\t\tClassification duration:  {classify_dur:.4g~#P}")
    logger.info(f"\t\tSegment closure duration: {close_segment_dur:.4g~#P}")
    logger.info(f"\t\tJ calculation duration:   {calculate_J_dur:.4g~#P}")
    logger.info(f"\t\tColor picking duration:   {color_picking_dur:.4g~#P}")

    return segments


# ====================== Segment Frequency Calculation ========================


def _calculate_segment_frequencies(
    segments: list[ContourSegment], data: dict, profile: Profile
):

    logger.info("==> Calculating frequencies...")

    # OPTIM: Switch to a non GUI backend because we are creating a lot of
    # contour plots. After the analysis is completed, we switch back to the
    # previous one. About +10% speed.
    default_backend = matplotlib.get_backend()
    matplotlib.use("ps")

    logger.disable("gcmotion")
    start = time()
    for i, seg in enumerate(
        tqdm(
            iterable=segments,
            desc=f"{'Calculating frequencies':^25}",
            unit=" freq",
            ascii=config.tqdm_style,
            colour=config.tqdm_color,
            disable=not config.pbar,
        )
    ):
        _calculate_omega_theta(seg, profile, data)
        _calculate_omega_zeta()
        _calculate_qkinetic()

        # Not closing the plots results in a memory leak, but closing them in
        # every loop has a huge performance impact. >20 seems good.
        if i % 40 == 0:
            plt.clf()
            plt.close()

    # Revert to previous backend
    matplotlib.use(default_backend)

    if not config.show_close_segments:
        # Do not show contours produced by frequency calculation
        plt.close()

    logger.enable("gcmotion")
    duration = Q(time() - start, "seconds")
    logger.info(f"\tSegments' frequencies calculated. Took {duration:.4g~#P}.")

    logger.enable("gcmotion")


def _calculate_omega_theta(
    segment: ContourSegment,
    profile: Profile,
    data: dict,
    **kwargs,
):
    r"""Calculates the ωθ of a single ContourSegment, independently from all
    the others."""

    E = segment.E
    Elower = E * (1 - config.energy_rtol)
    Eupper = E * (1 + config.energy_rtol)

    # Create a contour
    levels = (Elower, E, Eupper)
    C = plt.contour(
        "theta",
        "Ptheta",
        "Energy",
        data=data,
        levels=levels,
        algorithm="serial",
    )
    C.ylim = segment.ylim

    segs = tuple(_unpack_and_return_paths_optim(C=C, progress=False))

    upper_segs = tuple(seg for seg in segs if seg.E > E)
    lower_segs = tuple(seg for seg in segs if seg.E < E)

    upper_distances = tuple(
        segment.bbox_distance((seg.xmin, seg.ymin)) for seg in upper_segs
    )
    lower_distances = tuple(
        segment.bbox_distance((seg.xmin, seg.ymin)) for seg in lower_segs
    )

    # WARN: Might raise ValueError
    upper_seg = upper_segs[np.argmin(upper_distances)]
    lower_seg = lower_segs[np.argmin(lower_distances)]

    for seg in (upper_seg, lower_seg):
        seg.classify()
        seg.close_segment()
        seg.calculate_Jtheta()

    # Same as calculating the derivative from both sides and taking the mean.
    dE = upper_seg.E - lower_seg.E
    dJtheta = upper_seg.Jtheta - lower_seg.Jtheta

    segment.omega_theta = dE / dJtheta


def _calculate_omega_zeta():
    # TODO:
    pass


def _calculate_qkinetic():
    # TODO:
    pass


# ================================= Plots ===================================

# Manual lengend entries patches
trapped = Patch(color=config.trapped_color, label="Trapped")
copassing = Patch(color=config.copassing_color, label="Co-passing")
cupassing = Patch(color=config.cupassing_color, label="Counter-Passing")


def _plot_segments(segments: list[ContourSegment]):
    r"""Plots all segments."""

    if not config.show_segments:
        return

    logger.info("Plotting calculated segments...")

    fig = plt.figure(**global_fig_kw)
    ax = fig.add_subplot()

    # See LineCollection example
    # Unpacking and creating a LineCollection is MUCH faster.
    seg_vertices = tuple((np.column_stack(seg.vertices.T) for seg in segments))
    colors = tuple(seg.color for seg in segments)

    collection = LineCollection(seg_vertices, colors=colors)
    ax.add_collection(collection)
    ax.set_xlim([-np.pi, np.pi])
    ax.set_ylim(segments[1].ylim)
    ax.set_title(r"Contour Segments")
    ax.set_xlabel(r"$\theta [radians]$")
    ax.set_ylabel(r"$P_\theta [NU]$")

    ax.legend(handles=[trapped, copassing, cupassing], loc="upper right")
    plt.show()


def _plot_omega_thetas(segments):
    r"""Plots all omegas."""

    logger.info("Plotting omega thetas...")

    fig = plt.figure(**global_fig_kw)
    ax = fig.add_subplot()
    fig.suptitle(r"$\omega_\theta- Energy$")

    # See LineCollection example
    # Unpacking and creating a LineCollection is MUCH faster.
    omegas = tuple(seg.omega_theta for seg in segments)
    energies = tuple(seg.E for seg in segments)
    colors = tuple(seg.color for seg in segments)
    ax.scatter(energies, omegas, c=colors, s=config.scatter_size)

    ax.axhline(y=0, ls="--", lw=0.5, c="k")
    ax.set_xlabel("Energy [NU]")
    ax.set_ylabel(r"$\omega_\theta [\omega_0]$")

    ax.legend(handles=[trapped, copassing, cupassing])

    plt.show()


def _plot_omega_zetas(segments):

    logger.info("Plotting omega zetas...")

    fig = plt.figure(**global_fig_kw)
    ax = fig.add_subplot()
    fig.suptitle(r"$\omega_\zeta - Energy$")

    # See LineCollection example
    # Unpacking and creating a LineCollection is MUCH faster.
    omegas = tuple(seg.omega_zeta for seg in segments)
    energies = tuple(seg.E for seg in segments)
    colors = tuple(seg.color for seg in segments)
    ax.scatter(energies, omegas, c=colors, s=config.scatter_size)

    ax.set_xlabel("Energy [NU]")
    ax.set_ylabel(r"$\omega_\zeta [\omega_0]$")

    ax.legend(handles=[trapped, copassing, cupassing])

    plt.show()


def _plot_qkinetic(segments):

    logger.info("Plotting qkinetic...")

    fig = plt.figure(**global_fig_kw)
    ax = fig.add_subplot()
    fig.suptitle("qkinetic - Energy")

    # See LineCollection example
    # Unpacking and creating a LineCollection is MUCH faster.
    qs = tuple(seg.qkinetic for seg in segments)
    energies = tuple(seg.E for seg in segments)
    colors = tuple(seg.color for seg in segments)
    ax.scatter(energies, qs, c=colors, s=config.scatter_size)

    ax.set_xlabel("Energy [NU]")
    ax.set_ylabel(r"$q_{kin}$")

    ax.legend(handles=[trapped, copassing, cupassing])

    plt.show()
