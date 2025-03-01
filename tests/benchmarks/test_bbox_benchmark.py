import numpy as np
from numba import njit

vertices = np.random.random((1000000, 2))


def calculate_bbox() -> None:
    xmin, ymin = vertices.min(axis=0)
    xmax, ymax = vertices.max(axis=0)
    return ((xmin, ymin), (xmax, ymax))
    # xmin = vertices.T[0].min()
    # xmax = vertices.T[0].max()
    # ymin = vertices.T[1].min()
    # ymax = vertices.T[1].max()
    # return ((xmin, ymin), (xmax, ymax))


def calculate_bbox_transpose() -> None:
    xmin = vertices.T[0].min()
    xmax = vertices.T[0].max()
    ymin = vertices.T[1].min()
    ymax = vertices.T[1].max()
    return ((xmin, ymin), (xmax, ymax))


@njit(fastmath=True)
def calculate_bbox_jit() -> float:
    xmin = vertices.T[0].min()
    xmax = vertices.T[0].max()
    ymin = vertices.T[1].min()
    ymax = vertices.T[1].max()
    return ((xmin, ymin), (xmax, ymax))


# Call it once to compile
calculate_bbox_jit()


def test_calculate_bbox(benchmark):
    benchmark(calculate_bbox)


def test_calculate_bbox_transpose(benchmark):
    benchmark(calculate_bbox_transpose)


def test_calculate_bbox_jit(benchmark):
    benchmark(calculate_bbox_jit)


def test_bbox_benchmark_results(benchmark):
    assert calculate_bbox() == calculate_bbox_jit()
