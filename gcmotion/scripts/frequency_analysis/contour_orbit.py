import numpy as np
from math import isclose

from gcmotion.entities.profile import Profile
from gcmotion.configuration.scripts_configuration import ContourOrbitConfig

config = ContourOrbitConfig()
tau = 2 * np.pi


class ContourOrbit:
    r"""Path-like object containing the vertices as well as flags and the
    methods needed to classify the orbit.

    The methods should be called in a specific order, which is done inside
    profile_analysis() since some extra parameters are needed
    """

    vertices: np.ndarray = None
    E: float = None
    xmin: float = None
    ymin: float = None
    xmax: float = None
    ymax: float = None
    bbox: tuple[tuple[float, float], tuple[float, float]] = None

    valid: bool = None
    edge_orbit: bool = None
    passing: bool = None
    trapped: bool = None
    copassing: bool = None
    cupassing: bool = None
    area: float = None
    Jtheta: float = None
    Jzeta: float = None
    omega_theta: float = None
    omega_zeta: float = None
    qkinetic: float = None
    color: str = config.undefined_color

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

        # (bottom left point, top right point)
        self.bbox = ((self.xmin, self.ymin), (self.xmax, self.ymax))

    def validate(self, psilim: tuple) -> None:
        r"""Checks if the bbox of the contour line touches the upper or lower
        walls, which means the path gets cut off and must be discarded.
        """

        self.valid = is_inbounds(self, psilim) and not is_cutoff_trapped(self)

    def distance_from(self, bbox: tuple[tuple, tuple]) -> float:
        r"""Returns a distance-like quantity of the origin point from self's
        origin point. The closest bbox is the correct contour to calculate J.
        """
        return abs(self.xmin - bbox[0][0]) + abs(self.ymin - bbox[0][1])

    def classify_as_tp(self):
        r"""Classifies orbit as trapped or passing."""

        self.passing, self.trapped = tp_classify(self)

    def close_segment(self):
        r"""If the segment is passing, figure out if the points are
        left-to-right or right-to-left and append the two bottom points
        coorectly.

        Also append the first point to the end to close the orbit, even though
        it doesn't affect the shoelace algorithm.
        """
        if self.trapped:
            return

        left_to_right = is_left_to_right(self)

        closeoff_point = [self.vertices[0]]  # same for bot cases
        if left_to_right:
            extra = [[tau, 0], [-tau, 0]] + closeoff_point
            self.vertices = np.append(self.vertices, extra, axis=0)
        else:
            extra = [[-tau, 0], [tau, 0]] + closeoff_point
            self.vertices = np.append(self.vertices, extra, axis=0)

    def convert_to_ptheta(self, findPtheta: Profile, Q):
        r"""Converts all ycoords of the vertices from ψ to Pθ."""
        # Could not find a better way, but this isn't as slow as I thought.
        self.vertices = np.vstack(
            (
                self.vertices.T[0],  # Thetas as they were
                findPtheta(
                    Q(self.vertices.T[1], "NUMagnetic_flux"),
                    "NUCanonical_momentum",
                ).magnitude,
            )
        ).T

    def calculate_Jtheta(self):
        r"""Calculates the action J."""
        self.area = shoelace(self.vertices)
        if self.passing:
            self.area /= 2  # because theta span = 4π
        self.Jtheta = self.area / (2 * np.pi)

    def classify_as_cocu(self, profile: Profile):
        r"""Classifies orbit as co-/counter-passing."""
        self.undefined, self.copassing, self.cupassing = cocu_classify(
            self, profile
        )

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

    def str_dumb(self):

        # Use bool() to default to False if None
        tp = "t" * bool(self.trapped) + "p" * bool(self.passing)
        cocu = "co" * bool(self.copassing) + "cu" * bool(self.cupassing)
        edge = "/edge" * bool(self.edge_orbit)

        self.string = tp + "/" + cocu + edge


# ================================ Validation ================================


def is_inbounds(orbit: ContourOrbit, psilim: tuple) -> bool:
    r"""Checks if the path's bounding box is in bounds of the whole contour,
    e.g the contour line doesn't get cutoff.
    """
    # Must not be 0(default) when comparing with
    atol = config.inbounds_atol
    rtol = config.inbounds_rtol
    return not (
        isclose(
            orbit.ymin, psilim[0], abs_tol=atol, rel_tol=rtol
        )  # Touches floor
        or isclose(
            orbit.ymax, psilim[1], abs_tol=atol, rel_tol=rtol
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
