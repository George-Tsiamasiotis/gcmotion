import numpy as np
from math import isclose

from gcmotion.entities.profile import Profile
from gcmotion.configuration.scripts_configuration import ContourOrbitConfig

config = ContourOrbitConfig()
tau = 2 * np.pi


class ContourOrbit:
    r"""Path-like object containing vertices as well as flags and the methods
    needed to classify the orbit.

    The methods should be called in a specific order, which is done inside
    frequency_analysis() since some extra parameters are needed
    """

    def __init__(
        self,
        E: float,
        vertices: np.ndarray,
    ):

        self.vertices = vertices
        self.E = E

    def calculate_bbox(self) -> None:
        r"""Calculates the orbit's bounding box (smallest rectangle fully
        containing the orbit).

        Matplotlib's Path object does the exact same thing, so there is no
        reason to extend it.
        """

        self.xmin, self.ymin = self.vertices.min(axis=0)
        self.xmax, self.ymax = self.vertices.max(axis=0)

    def validate(self, ylim: tuple) -> None:
        r"""Checks if the bbox of the contour line touches the upper or lower
        walls, which means the path gets cut off and must be discarded.
        """

        self.valid = is_inbounds(self, ylim) and not is_cutoff_trapped(self)

    def classify(self, profile: Profile = None):
        r"""Classifies the segment to trapped/passing and left-to-right, needed
        to correctly add the bottom points in the correct order.

        Since the same class is used to create 'phony orbits' to calclulate dE
        and dJtheta localy, co-/counter-passing classification is not
        necessary.
        """

        self.passing, self.trapped = tp_classify(self)
        self.left_to_right = is_left_to_right(self)
        if profile is None:
            return
        if self.trapped:
            return
        self.undefined, self.copassing, self.cupassing = cocu_classify(
            self, profile
        )

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
        if getattr(self, "undefined", False):
            self.color = config.undefined_color
            return

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


def is_inbounds(orbit: ContourOrbit, ylim: tuple) -> bool:
    r"""Checks if the path's bounding box is in bounds of the whole contour,
    e.g the contour line doesn't get cutoff.
    """
    # Must not be 0(default) when comparing with
    atol = config.inbounds_atol
    rtol = config.inbounds_rtol
    return not (
        isclose(
            orbit.ymin, ylim[0], abs_tol=atol, rel_tol=rtol
        )  # Touches floor
        or isclose(
            orbit.ymax, ylim[1], abs_tol=atol, rel_tol=rtol
        )  # Touches ceil
    )


def is_cutoff_trapped(orbit: ContourOrbit) -> bool:
    r"""Checks if the segment is cut off by left or the right walls.

    Checks that the bounding box doesn't touch the left *and* the right wall
    (e.g. not passing), but touches at least one of them. This means that it is
    a trapped orbit that gets cutoff by the contour. This orbit is reduntant
    since it will be found at θ=0 too.
    """
    return (isclose(orbit.xmin, -tau) and not isclose(orbit.xmax, tau)) or (
        isclose(orbit.xmax, tau) and not isclose(orbit.xmin, -tau)
    )


# ===================== Trapped - Passing Classification =====================


def tp_classify(path: ContourOrbit) -> list[bool, bool]:
    r"""Checks the left and right bbox edges to check if passing or trapped."""

    # NOTE: isclose() is needed here
    if isclose(path.xmin, -tau) and isclose(path.xmax, tau):
        return True, False
    else:
        return False, True


def cocu_classify(path: ContourOrbit, profile: Profile) -> list[bool, bool]:
    r"""Classifies the segment as co-passing or counter-passing depending on
    the sign of rho"""
    # psis = path.vertices.T[1]
    # sample_idx = np.round(
    #     np.linspace(1, len(psis) - 1, config.rho_sample_size)
    # ).astype(int)
    # co = profile._rhosign(psis[sample_idx])

    undefined, co = profile._rhosign(psiNU=path.vertices.T[1])

    return undefined, co, not co


# ======================= Left-to-Right Classification =======================


def is_left_to_right(path: ContourOrbit) -> bool:
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
