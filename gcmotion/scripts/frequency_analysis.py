import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from tqdm import tqdm
from time import time
from statistics import mean
from dataclasses import asdict
from collections import deque
from matplotlib.contour import QuadContourSet
from matplotlib.patches import Rectangle

from gcmotion.utils.logger_setup import logger
from gcmotion.entities.profile import Profile
from gcmotion.configuration.scripts_configuration import ContourFreqConfig
from gcmotion.scripts.utils.contour_segment import ContourSegment
from gcmotion.plot._base._base_profile_energy_contour import (
    _base_profile_energy_contour,
)

config = ContourFreqConfig()
matplotlib.rcParams["savefig.dpi"] = 250
fig_kw = {
    "figsize": config.figsize,
    "dpi": config.dpi,
    "layout": config.layout,
}


def frequency_analysis(profile: Profile, psilim: list, **kwargs):

    # Unpack it here and updated, since only this function is to be called
    # externaly
    for key, value in kwargs.items():
        setattr(config, key, value)

    logger.info("Creating master Contour...")
    C = _contour_set(profile, psilim)

    logger.info("Unpacking paths...")
    segs = _unpack_paths(C=C, progress=config.pbar)

    logger.info("Validating Segments...")
    segs = _validate_and_return(segments=segs)

    logger.info("Preparing Segments...")
    segs = _prepare_segments(segments=segs)

    if config.show_segments:
        _plot_segments(segs)

    _calculate_segment_frequencies(segs, C.data)

    if not config.show_close_segments:
        # Do not show contours produced by frequency calculation
        plt.close()

    segs = [seg for seg in segs if hasattr(seg, "omega_theta")]

    _plot_omegas(segments=segs)

    return segs


def _contour_set(profile: Profile, psilim: list, **kwargs):

    fig = plt.figure(**fig_kw)
    ax = fig.add_subplot()

    # Override default parameters
    # All operations are done in NU.
    contour_kw = {
        **asdict(config),
        "thetalim": [-2 * np.pi, 2 * np.pi],
        "psilim": psilim,
        "levels": config.levels,
        "grid_density": config.grid_density,
        "ycoord": "Ptheta",
        "E_units": "NUJoule",
        "flux_units": "NUMagnetic_flux",
        "canomom_units": "NUCanonical_momentum",
        "locator": config.locator,
        "log_base": config.log_base,
        "mode": "lines",
    }
    C = _base_profile_energy_contour(profile, ax=ax, **contour_kw)
    logger.info(f"Found {len(C.allsegs)} Contours.")

    if config.show_master_contour:
        plt.title("Master Contour")
        plt.show()
    else:
        plt.close()

    return C


def _unpack_paths(C: QuadContourSet, progress: bool = True) -> None:
    r"""Iterates over C.get_paths(), unpacking each one and creating a
    ContourSegment, assigning to it the correct energy. It also discards all
    empty segments, which are unavoidably yielded from the gererator.

    Returns
    -------
    list
        A list containing *all* the extracted ContourSegments

    """

    segments = deque()

    # Iterate over allsegs and create ContourSegments
    logger.info("Extracting ContourSegments")
    for E, composite_path in tqdm(
        iterable=zip(C.levels, C.get_paths()),
        desc=f"{"Creating ContourSegments":^25}",
        ascii=config.tqdm_style,
        unit=" segs",
        disable=not progress,
    ):
        # HACK: C.allsegs does the exact same thing, though we can't surely
        # assign the correct energy to each seg. However, this method also
        # relies on the contour object return the Paths in rising energy
        # levels, since C.levels is sorted.

        segment_genenerator = composite_path._iter_connected_components()

        for segment in segment_genenerator:
            if segment.vertices.shape == (0, 2):  # Discard empty segments
                continue

            new_segment = ContourSegment(segment=segment, E=E, ylim=C.ylim)
            segments.append(new_segment)

    return segments


# ====================== ContourSegment Instantiation ========================


def _validate_and_return(
    segments: list[ContourSegment],
) -> list[ContourSegment]:
    r"""Makes every segment validate itself and returns the valid ones.

    Returns
    -------
    list
        A list containing all the valid ContourSegments

    """

    for seg in segments:
        seg.validate()

    valid = [seg for seg in segments if seg.valid]

    logger.info(f"Kept {len(valid)}/{len(segments)} valid segments.")

    return valid


def _prepare_segments(
    segments: list[ContourSegment],
) -> list[ContourSegment]:
    r"""Makes all ContourSegments classify themselves"""

    for seg in segments:
        seg.classify()
        seg.close_segment()
        seg.calculate_J()
        if config.del_vertices and not config.show_segments:
            delattr(seg, "vertices")

    return segments


# ====================== Segment Frequency Calculation ========================


def _calculate_segment_frequencies(segments: list[ContourSegment], data: dict):

    logger.disable("gcmotion")
    for seg in tqdm(
        iterable=segments,
        desc=f"{'Calculating frequencies':^25}",
        ascii=config.tqdm_style,
        unit=" freq",
        disable=not config.pbar,
    ):
        _calculate_segment_frequency(seg, data)

    logger.enable("gcmotion")


def _calculate_segment_frequency(
    segment: ContourSegment,
    data: dict,
    **kwargs,
):
    r"""Calculates the ωθ of a single ContourSegment, independently from all
    the others."""
    if not segment.valid:
        return

    E = segment.E
    Elower = E * (1 - config.energy_rtol)
    Eupper = E * (1 + config.energy_rtol)

    # Create a contour
    C = plt.contour(
        "theta", "ycoord", "Energy", data=data, levels=[Elower, E, Eupper]
    )
    C.ylim = segment.ylim
    segs = _unpack_paths(C=C, progress=False)

    upper_segs = [seg for seg in segs if seg.E > E]
    lower_segs = [seg for seg in segs if seg.E < E]

    upper_segs = _validate_and_return(upper_segs)
    upper_segs = _prepare_segments(upper_segs)
    lower_segs = _validate_and_return(lower_segs)
    lower_segs = _prepare_segments(lower_segs)

    upper_distances = [
        segment.bbox_distance((seg.xmin, seg.ymin)) for seg in upper_segs
    ]
    lower_distances = [
        segment.bbox_distance((seg.xmin, seg.ymin)) for seg in lower_segs
    ]

    try:
        upper_seg = upper_segs[np.argmin(upper_distances)]
        lower_seg = lower_segs[np.argmin(lower_distances)]
    except ValueError:
        logger.trace("Skipped segment")
        return

    # Sanity check
    assert lower_seg.E == Elower
    assert upper_seg.E == Eupper

    dE1 = upper_seg.E - segment.E
    dE2 = segment.E - lower_seg.E
    dE = mean([dE1, dE2])

    dJ1 = upper_seg.J - segment.J
    dJ2 = segment.J - lower_seg.J
    dJ = mean([dJ1, dJ2])

    segment.omega_theta = dE / dJ


# ================================= Plots ===================================


def _plot_segments(segments: list[ContourSegment]):
    r"""Plots all segments."""

    fig = plt.figure(**fig_kw)
    ax = fig.add_subplot()

    for seg in segments:
        x, y = seg.vertices.T[:]
        ax.plot(x, y, color=seg.color)

    plt.show()


def _plot_omegas(segments):

    fig = plt.figure(**fig_kw)
    ax = fig.add_subplot()

    for seg in segments:
        ax.scatter(seg.E, seg.omega_theta, c=seg.color, s=config.scatter_size)

    ax.set_xlabel("Energy [NU]")
    ax.set_ylabel(r"$\omega [\omega_0]$")

    plt.show()
