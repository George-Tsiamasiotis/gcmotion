import pytest
import numpy as np
from numba import njit

pytestmark = pytest.mark.benchmark(group="shoelace")


# @pytest.fixture(scope="module")
def shoelace(x, y) -> float:
    return float(
        0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    )


# @pytest.fixture(scope="module")
@njit("float64(float64[::1], float64[::1])")
def shoelace_jit(x, y) -> float:
    return float(
        0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    )


x, y = (*np.random.random((2, 1000)),)

# Call it once to compile
_ = shoelace_jit(x=x, y=y)


def test_shoelace(benchmark):
    benchmark(shoelace, x=x, y=y)


def test_shoelace_jit(benchmark):
    benchmark(shoelace_jit, x=x, y=y)
