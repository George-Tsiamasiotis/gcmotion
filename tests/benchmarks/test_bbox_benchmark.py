import pytest
import numpy as np

from gcmotion.scripts.frequency_analysis.contour_orbit import _calculate_bbox

pytestmark = pytest.mark.benchmark(group="bbox")

vertices = np.random.random((10000, 2))


def calculate_bbox_normal(vertices) -> None:
    xmin, ymin = vertices.min(axis=0)
    xmax, ymax = vertices.max(axis=0)
    return xmin, ymin, xmax, ymax


def calculate_bbox_transpose(vertices) -> None:
    xmin = vertices.T[0].min()
    xmax = vertices.T[0].max()
    ymin = vertices.T[1].min()
    ymax = vertices.T[1].max()
    return xmin, ymin, xmax, ymax


def test_calculate_bbox_normal(benchmark):
    benchmark(calculate_bbox_normal, vertices)


def test_calculate_bbox_transpose(benchmark):
    benchmark(calculate_bbox_transpose, vertices)


def test_calculate_bbox_jit(benchmark):
    benchmark(_calculate_bbox, *vertices.T)


def test_bbox_benchmark_results():
    assert np.all(
        np.isclose(
            calculate_bbox_normal(vertices), _calculate_bbox(*vertices.T)
        )
    )
