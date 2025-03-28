from math import isclose
import numpy as np

# ====================TESTING FUNCTIONS=============================


def test_parabolas_default_instantiation(parabolas_output):
    r"""Tests instantiation essentally."""
    parabolas_output = parabolas_output

    x = parabolas_output["x"]
    x_TPB = parabolas_output["x_TPB"]
    y_R = parabolas_output["y_R"]
    y_L = parabolas_output["y_L"]
    y_MA = parabolas_output["y_MA"]
    TPB_O = parabolas_output["TPB_O"]
    TPB_X = parabolas_output["TPB_X"]
    v_R = parabolas_output["v_R"]
    v_L = parabolas_output["v_L"]
    v_MA = parabolas_output["v_MA"]

    assert len(x) == len(y_R) == len(y_L) == len(y_MA) == 1000
    assert len(x_TPB) == len(TPB_O) == len(TPB_X) == 1000
    assert len(v_L) == len(v_R) == len(v_MA) == 2


def test_parabolas_verticies(parabolas_output):
    r"""Tests if the parabolas' verticies are calculated correctly."""
    parabolas_output = parabolas_output

    v_R = parabolas_output["v_R"]
    v_L = parabolas_output["v_L"]
    v_MA = parabolas_output["v_MA"]

    assert isclose(v_R[0], -1, abs_tol=1e-3)
    assert isclose(v_L[0], -1, abs_tol=1e-3)
    assert v_MA[0] <= 1e-3
    assert v_R[1] <= v_MA[1] <= v_L[1]


def test_parabolas_RW_LW(parabolas_output):
    r"""Tests correct calculations of wall (boundary) parabolas."""
    parabolas_output = parabolas_output

    y_R = parabolas_output["y_R"]
    y_L = parabolas_output["y_L"]

    assert np.all(y_L >= y_R)


def test_parabolas_tpb_at_zero(parabolas_output):
    r"""Tests correct calculaion of TP-boundary near 0."""
    parabolas_output = parabolas_output

    TPB_O = parabolas_output["TPB_O"]
    TPB_X = parabolas_output["TPB_X"]

    assert isclose(TPB_O[-1], TPB_X[-1], abs_tol=5e-3)
