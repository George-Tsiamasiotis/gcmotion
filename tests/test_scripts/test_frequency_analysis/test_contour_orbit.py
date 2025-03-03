r"""
=========
IMPORTANT
=========

Deepcopy and rerun all needed methods in each test to make the tests
independent from eachother.
"""

import pytest
import numpy as np

from math import isclose
from copy import deepcopy
from dataclasses import dataclass

from gcmotion.scripts.frequency_analysis.contour_orbit import (
    ContourOrbit,
    is_inbounds,
    is_cutoff_trapped,
    is_left_to_right,
)
from gcmotion.configuration.scripts_configuration import ContourOrbitConfig

tau = 2 * np.pi

psilim = (0, 1)


@pytest.fixture(scope="module")
def trapped_contour_orbit():
    r"""Creates a circlular orbit around (0, 0.5) with radius 0.25 and E=1e-5,
    assuming that psi_wall=1.
    """
    t = np.linspace(0, tau, 2000)
    x = 0.25 * np.cos(t)
    y = 0.5 + 0.25 * np.sin(t)
    vertices = np.array((x, y)).T

    yield ContourOrbit(vertices=vertices, E=1e-5)


@pytest.fixture(scope="module")
def passing_contour_orbit():
    r"""Creates a passing contour orbit represented as a sinewave between
    [-tau, tau] with 'DC bias' of 0.5 and amplitude 0.25, assuming psi_wall=1.
    """
    x = np.linspace(-tau, tau, 2000)
    y = 0.25 + 0.5 * np.sin(0.5 * x) ** 2
    vertices = np.array((x, y)).T

    yield ContourOrbit(vertices=vertices, E=1e-5)


@pytest.fixture(scope="module")
def cutoff_trapped_contour_orbit():
    r"""Creates a half-circlular orbit around (2π, 0.5) with radius 0.25 and
    E=1e-5, assuming that psi_wall=1.:w.
    """
    t = np.linspace(np.pi / 2, 3 * np.pi / 2, 200)
    x = tau + 0.25 * np.cos(t)
    y = 0.5 + 0.25 * np.sin(t)
    vertices = np.array((x, y)).T

    yield ContourOrbit(vertices=vertices, E=1e-5)


# =============================================================================


def test_validate(
    trapped_contour_orbit,
    passing_contour_orbit,
    cutoff_trapped_contour_orbit,
):
    r"""Only test functionality, since its parts get tested later"""
    trapped_orbit = deepcopy(trapped_contour_orbit)
    passing_orbit = deepcopy(passing_contour_orbit)
    cutoff_orbit = deepcopy(cutoff_trapped_contour_orbit)

    global psilim
    trapped_orbit.calculate_bbox()
    passing_orbit.calculate_bbox()
    cutoff_orbit.calculate_bbox()

    trapped_orbit.validate(psilim)
    passing_orbit.validate(psilim)
    cutoff_orbit.validate(psilim)

    assert trapped_orbit.valid
    assert passing_orbit.valid
    assert not cutoff_orbit.valid


def test_trapped_contour_orbit_bbox(trapped_contour_orbit):

    orbit = deepcopy(trapped_contour_orbit)

    orbit.calculate_bbox()
    bbox = orbit.bbox

    # 1% absolute tolerance is much more than we will ever need
    assert isclose(bbox[0][0], -0.25, abs_tol=tau / 100)  # xmin
    assert isclose(bbox[1][0], 0.25, abs_tol=tau / 100)  # xmax
    assert isclose(bbox[0][1], 0.25, abs_tol=1 / 100)  # ymin
    assert isclose(bbox[1][1], 0.75, abs_tol=1 / 100)  # ymax


def test_passing_contour_orbit_bbox(passing_contour_orbit):

    orbit = deepcopy(passing_contour_orbit)

    orbit.calculate_bbox()
    bbox = orbit.bbox

    # 1% absolute tolerance is much more than we will ever need
    assert isclose(bbox[0][0], -tau, abs_tol=tau / 100)  # xmin
    assert isclose(bbox[1][0], tau, abs_tol=tau / 100)  # xmax
    assert isclose(bbox[0][1], 0.25, abs_tol=1 / 100)  # ymin
    assert isclose(bbox[1][1], 0.75, abs_tol=1 / 100)  # ymax


def test_is_inbounds():

    global psilim

    # Phony class to hold the attributes ymin and ymax as expected by
    # is_inbounds()
    @dataclass
    class PhonyOrbit:
        ymin: float
        ymax: float

    inbounds1 = PhonyOrbit(ymin=0.2, ymax=0.6)
    outofbounds1 = PhonyOrbit(ymin=0, ymax=0.6)
    outofbounds2 = PhonyOrbit(ymin=0.2, ymax=1)
    outofbounds3 = PhonyOrbit(ymin=-0.2, ymax=0.6)
    outofbounds4 = PhonyOrbit(ymin=0.2, ymax=1.6)
    outofbounds5 = PhonyOrbit(ymin=-0.2, ymax=1.6)

    assert is_inbounds(inbounds1, psilim)
    assert not is_inbounds(outofbounds1, psilim)
    assert not is_inbounds(outofbounds2, psilim)
    assert not is_inbounds(outofbounds3, psilim)
    assert not is_inbounds(outofbounds4, psilim)
    assert not is_inbounds(outofbounds5, psilim)


def test_is_cutoff_trapped(
    trapped_contour_orbit,
    passing_contour_orbit,
    cutoff_trapped_contour_orbit,
):
    r"""Since we dont expect to encounter any orbit that cannot be described as
    one of the 3 fixtures, its enough to test only those."""
    trapped_orbit = deepcopy(trapped_contour_orbit)
    passing_orbit = deepcopy(passing_contour_orbit)
    cutoff_orbit = deepcopy(cutoff_trapped_contour_orbit)
    trapped_orbit.calculate_bbox()
    passing_orbit.calculate_bbox()
    cutoff_orbit.calculate_bbox()

    assert not is_cutoff_trapped(trapped_orbit)
    assert not is_cutoff_trapped(passing_orbit)
    assert is_cutoff_trapped(cutoff_orbit)


def test_distance_from(
    trapped_contour_orbit,
    passing_contour_orbit,
):
    r"""distance_from() returns a distance-like object which is too simple to
    test, just make sure it works."""
    trapped_orbit = deepcopy(trapped_contour_orbit)
    passing_orbit = deepcopy(passing_contour_orbit)
    bbox = ((-2, 0.25), (2, 0.75))

    trapped_orbit.calculate_bbox()
    passing_orbit.calculate_bbox()
    trapped_orbit.distance_from(bbox)
    passing_orbit.distance_from(bbox)


def test_trapped_passing_classification(
    trapped_contour_orbit,
    passing_contour_orbit,
):
    trapped_orbit = deepcopy(trapped_contour_orbit)
    passing_orbit = deepcopy(passing_contour_orbit)

    trapped_orbit.calculate_bbox()
    passing_orbit.calculate_bbox()
    trapped_orbit.classify_as_tp()
    passing_orbit.classify_as_tp()

    assert trapped_orbit.trapped
    assert passing_orbit.passing


def test_close_segment(trapped_contour_orbit, passing_contour_orbit):
    trapped_orbit = deepcopy(trapped_contour_orbit)
    passing_orbit = deepcopy(passing_contour_orbit)

    global psilim
    trapped_orbit.calculate_bbox()
    trapped_orbit.validate(psilim)
    trapped_orbit.classify_as_tp()
    passing_orbit.calculate_bbox()
    passing_orbit.validate(psilim)
    passing_orbit.classify_as_tp()

    trapped_orbit.close_segment()
    passing_orbit.close_segment()

    # Test that they are the same for trapped
    assert np.all(trapped_orbit.vertices == trapped_contour_orbit.vertices)
    assert (
        passing_orbit.vertices.shape[0]
        == passing_contour_orbit.vertices.shape[0] + 3
    )
    assert np.all(
        passing_orbit.vertices[:-3] == passing_contour_orbit.vertices
    )


def test_is_left_to_right(passing_contour_orbit):

    passing_orbit = deepcopy(passing_contour_orbit)
    inv_orbit = ContourOrbit(vertices=passing_orbit.vertices[::-1], E=1e-5)

    # Must be run before close_segment()
    assert is_left_to_right(passing_orbit)
    assert not is_left_to_right(inv_orbit)

    passing_orbit.close_segment()
    inv_orbit.close_segment()

    # Check last 2 points
    assert np.all(passing_orbit.vertices[-2] == np.array([-tau, 0]))
    assert np.all(inv_orbit.vertices[-2] == np.array([tau, 0]))
    assert np.all(passing_orbit.vertices[-1] == np.array([-tau, 0.25]))
    assert np.all(inv_orbit.vertices[-1] == np.array([tau, 0.25]))


def test_convert_to_ptheta(
    simple_profile, trapped_contour_orbit, passing_contour_orbit
):
    r"""Just test functionality"""
    #  trapped_contour_orbit.
    trapped_orbit = deepcopy(trapped_contour_orbit)
    passing_orbit = deepcopy(passing_contour_orbit)

    trapped_orbit.convert_to_ptheta(
        simple_profile.findPtheta, simple_profile.Q
    )
    passing_orbit.convert_to_ptheta(
        simple_profile.findPtheta, simple_profile.Q
    )


def test_area_calculation(
    simple_profile, trapped_contour_orbit, passing_contour_orbit
):
    r"""Without converting to Ptheta, test that the area is correct."""
    #  trapped_contour_orbit.
    trapped_orbit = deepcopy(trapped_contour_orbit)
    passing_orbit = deepcopy(passing_contour_orbit)

    global psilim
    trapped_orbit.calculate_bbox()
    trapped_orbit.validate(psilim)
    trapped_orbit.classify_as_tp()
    trapped_orbit.close_segment()
    passing_orbit.calculate_bbox()
    passing_orbit.validate(psilim)
    passing_orbit.classify_as_tp()
    passing_orbit.close_segment()

    trapped_orbit.calculate_Jtheta()
    passing_orbit.calculate_Jtheta()

    trapped_area = np.pi * 0.25**2
    assert isclose(trapped_orbit.area, trapped_area, rel_tol=1 / 10000)

    # From numerical integration:
    passing_area = 6.283185
    # Divide by 2 since we need the area inside [-π,π]
    assert isclose(passing_orbit.area, passing_area / 2, rel_tol=1 / 10000)


def test_color():

    # Phony class to hold the attributes ymin and ymax as expected by
    # is_inbounds()
    vertices = np.random.random((4, 2))
    config = ContourOrbitConfig()

    trapped = ContourOrbit(vertices=vertices, E=1)
    copassing = ContourOrbit(vertices=vertices, E=1)
    cupassing = ContourOrbit(vertices=vertices, E=1)
    undefined = ContourOrbit(vertices=vertices, E=1)

    trapped.trapped = True
    copassing.copassing = True
    cupassing.cupassing = True
    undefined.undefined = True

    trapped.pick_color()
    copassing.pick_color()
    cupassing.pick_color()
    undefined.pick_color()

    assert trapped.color == config.trapped_color
    assert copassing.color == config.copassing_color
    assert cupassing.color == config.cupassing_color
    assert undefined.color == config.undefined_color


def test_str_dump(trapped_contour_orbit, passing_contour_orbit):
    # TODO: For now test only functionality until i decide the final format.
    trapped_contour_orbit.str_dump()
    passing_contour_orbit.str_dump()
