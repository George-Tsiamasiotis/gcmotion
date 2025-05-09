import numpy as np
import pytest
from math import isclose, pi

from gcmotion.scripts.frequency_analysis.contour_orbit import shoelace


@pytest.fixture(scope="module")
def _unit_square():
    r"""Creates a np.array with the points of the unit square."""
    x = [0, 0, 1, 1]
    y = [0, 1, 1, 0]
    return np.array([x, y]).astype(float)


@pytest.fixture(scope="module")
def _unit_square_n():
    r"""Creates a np.array with the points of the unit square, backwards."""
    x = [0, 1, 1, 0]
    y = [0, 0, 1, 1]
    return np.array([x, y]).astype(float)


@pytest.fixture(scope="module")
def _unit_circle():
    r"""Creates a np.array with 1000 points in the unit circle."""
    t = np.linspace(0, 2 * pi, 1000)
    x = np.cos(t)
    y = np.sin(t)
    return np.array([x, y])


@pytest.fixture(scope="module")
def _unit_circle_n():
    r"""Creates a np.array with 1000 points in the unit circle, backwards."""
    t = np.linspace(0, 2 * pi, 1000)
    x = -np.cos(t)
    y = -np.sin(t)
    return np.array([x, y])


@pytest.fixture(scope="module")
def _unit_half_circle():
    r"""Creates a np.array with 1000 points in the unit circle."""
    t = np.linspace(0, pi, 1000)
    x = np.cos(t)
    y = np.sin(t)
    return np.array([x, y])


@pytest.fixture(scope="module")
def _unit_half_circle_n():
    r"""Creates a np.array with 1000 points in the unit circle, backwards."""
    t = np.linspace(0, pi, 1000)
    x = -np.cos(t)
    y = -np.sin(t)
    return np.array([x, y])


def test_shoelace(
    _unit_square,
    _unit_circle,
    _unit_square_n,
    _unit_circle_n,
    _unit_half_circle,
    _unit_half_circle_n,
):

    # Positives
    assert isclose(shoelace(*_unit_square), 1, abs_tol=0.01)
    assert isclose(shoelace(*_unit_circle), pi, abs_tol=0.01)
    assert isclose(shoelace(*_unit_half_circle), pi / 2, abs_tol=0.01)
    # Negatives
    assert isclose(shoelace(*_unit_square_n), 1, abs_tol=0.01)
    assert isclose(shoelace(*_unit_circle_n), pi, abs_tol=0.01)
    assert isclose(shoelace(*_unit_half_circle_n), pi / 2, abs_tol=0.01)
    pass
