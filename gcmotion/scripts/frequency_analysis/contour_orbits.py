import numpy as np
from math import isclose
from matplotlib.path import Path

from gcmotion.entities.profile import Profile
from gcmotion.configuration.scripts_configuration import ContourFreqConfig

config = ContourFreqConfig()
tau = 2 * np.pi


class ContourOrbit(Path):

    def __init__(self, E: float, ylim: tuple = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.E = E
        self.ylim = ylim

        # Bounding box
        (self.xmin, self.ymin), (self.xmax, self.ymax) = (
            self.get_extents().get_points()
        )

    def validate(self) -> None:
        r"""Checks if the bbox of the contour line touches the upper or lower
        walls, which means the path gets cut off and must be discarded.
        """
        self.valid = _is_inbounds(self) and not _is_cutoff_trapped(self)

    def classify(self, profile: Profile = None):
        r"""Classifies the segment to trapped/passing and left-to-right, needed
        to correctly add the bottom points.
        """
        self.passing, self.trapped = _tp_classify(self)
        self.left_to_right = _is_left_to_right(self)
        if profile is None:
            return
        self.copassing, self.cupassing = _cocu_classify(self, profile)

    def close_segment(self):
        r"""If the segment is passing, append the two bottom points, as well as
        the first point to close the segment. Not needed for the shoelace
        algorithm, but makes plotting clearer.
        """
        if self.trapped:
            return

        closeoff_point = [self.vertices[0]]  # same for bot cases
        if self.left_to_right:
            extra = [[tau, 0], [-tau, 0]] + closeoff_point
            self.vertices = np.append(self.vertices, extra, axis=0)
        else:
            extra = [[-tau, 0], [tau, 0]] + closeoff_point
            self.vertices = np.append(self.vertices, extra, axis=0)

    def calculate_Jtheta(self):
        r"""Calculates the action J."""
        area = shoelace(self.vertices)
        if self.passing:
            area /= 2  # because theta span = 4π
        self.Jtheta = area / (2 * np.pi)

    def distance_from(self, xy: tuple[float, float]) -> float:
        r"""Returns a distance-like quantity of the origin point from self's
        origin point. The closest bbox is the correct contour to calculate J.
        """
        return abs(self.xmin - xy[0]) + abs(self.ymin - xy[1])

    def pick_color(self):
        r"""Sets the segment's color depending on its orbit type."""
        # TODO: find a better way to do this
        self.color = (
            config.trapped_color
            if self.trapped
            else (
                config.copassing_color
                if self.copassing
                else (
                    config.cupassing_color
                    if self.cupassing
                    else config.undefined_color
                )
            )
        )


# ================================ Validation ================================


def _is_inbounds(path: ContourOrbit) -> bool:
    r"""Checks if the path's bounding box is in bounds of the whole contour,
    e.g the contour line doesn't get cutoff.
    """
    # Must not be 0(default) when comparing with
    atol = config.is_inbounds_atol
    return not (
        isclose(path.ymin, path.ylim[0], abs_tol=atol)  # Touches floor
        or isclose(path.ymax, path.ylim[1], abs_tol=atol)  # Touches ceil
    )


def _is_cutoff_trapped(path: ContourOrbit) -> bool:
    r"""Checks if the segment is cut off by left or the right walls.

    Checks that the bounding box doesn't touch the left *and* the right wall
    (e.g. not passing), but touches at least one of them. This means that it is
    a trapped orbit that gets cutoff by the contour. This orbit is reduntant
    since it will be found at θ=0 too.
    """
    return (isclose(path.xmin, -tau) and not isclose(path.xmax, tau)) or (
        isclose(path.xmax, tau) and not isclose(path.xmin, -tau)
    )


# ===================== Trapped - Passing Classification =====================


def _tp_classify(path: ContourOrbit) -> list[bool, bool]:
    r"""Checks the left and right bbox edges to check if passing or trapped."""

    # NOTE: isclose() is needed here
    if isclose(path.xmin, -tau) and isclose(path.xmax, tau):
        return True, False
    else:
        return False, True


def _cocu_classify(path: ContourOrbit, profile: Profile) -> list[bool, bool]:
    r"""Classifies the segment as co-passing or counter-passing depending on
    the sign of rho"""
    # OPTIM: After some testing, we can see that for passing particles, the rho
    # variable is always either negative or positive. For the shake of
    # optimization, it is safe to assume that if some evenly spaced values of
    # psi correspond to negative rho, then all of them do. The sample size can
    # be tweaked in the configuration

    psis = path.vertices.T[1]
    sample_idx = np.round(
        np.linspace(1, len(psis) - 1, config.rho_sample_size)
    ).astype(int)
    co = profile._rhosign(psis[sample_idx])

    return co, not co


# ======================= Left-to-Right Classification =======================


def _is_left_to_right(path: ContourOrbit) -> bool:
    r"""In the case that the path is passing, it returns True if the *first*
    point touches the left wall. It also checks that the *last* point touches
    the third wall, which might be reduntant."""
    if path.trapped:
        return None

    # NOTE: isclose() is needed here
    return (
        True
        if (
            isclose(path.vertices[0][0], -tau)
            and isclose(path.vertices[-1][0], tau)
        )
        else False
    )


# ================================= Shoelace =================================


def shoelace(array: np.ndarray) -> float:
    r"""Calculates the area of a polygon.

    It is not necessary that the polygon is closed. The algorithm assumes the
    first and last points are connected.

    Parameters
    ----------
    array: np.ndarray
        (N,2) numpy array containing the polygon's points.

    Returns
    -------
    float
        The area of the polygon.

    """
    x, y = (*array.T,)
    return float(
        0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    )
