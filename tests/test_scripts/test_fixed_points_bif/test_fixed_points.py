import numpy as np
from math import isclose

from gcmotion.scripts.fixed_points_bif.XO_points_classification import (
    XO_points_classification as xoc,
)


# ======================TESTING FUNCTIONS========================


def test_fixed_points_efield(fp_output, giorgos_efield_profile):
    r"""Tests/Compares fixed points' theta, psi coordinates
    against verified results."""

    fixed_points_output = fp_output

    fixed_points = fixed_points_output["distinct_fixed_points"]

    # Classify fixed points to X O Points
    X_points, O_points = xoc(
        unclassified_fixed_points=fixed_points,
        profile=giorgos_efield_profile,
        to_P_thetas=False,
    )

    # Convert deque to numpy arrays for easy manipulation
    X_thetas, X_psisNU = zip(*X_points) if X_points else ([], [])
    O_thetas, O_psisNU = zip(*O_points) if O_points else ([], [])

    X_psis = (
        giorgos_efield_profile.Q(X_psisNU, "NUMagnetic_flux").to("psi_wall").m
    )
    O_psis = (
        giorgos_efield_profile.Q(O_psisNU, "NUMagnetic_flux").to("psi_wall").m
    )

    pi = np.pi

    assert isclose(X_thetas[0], pi, abs_tol=1e-10)
    assert X_thetas[1] < 1e-10
    assert isclose(X_thetas[2], pi, abs_tol=1e-10)

    assert O_thetas[0] < 1e-10
    assert isclose(O_thetas[1], pi, abs_tol=1e-10)
    assert O_thetas[2] < 1e-10

    assert isclose(X_psis[0], 0.940517, abs_tol=1e-5)
    assert isclose(X_psis[1], 1.012314, abs_tol=1e-5)
    assert isclose(X_psis[2], 1.173791, abs_tol=1e-5)

    assert isclose(O_psis[0], 0.918363, abs_tol=1e-5)
    assert isclose(O_psis[1], 0.988851, abs_tol=1e-5)
    assert isclose(O_psis[2], 1.211453, abs_tol=1e-5)


def test_fixed_points_init_conds(fp_output):
    r"""Tests that no more fixed points where found than there where
    initial conditions."""
    fixed_points_output = fp_output

    fixed_points = fixed_points_output["distinct_fixed_points"]
    initial_conditions = fixed_points_output["initial_conditions"]

    assert len(fixed_points) <= len(initial_conditions)


def test_fixed_points_type(fp_output, giorgos_efield_profile):
    r"""Tests X O Points classification algorithm essentially."""
    fixed_points_output = fp_output

    fixed_points = fixed_points_output["distinct_fixed_points"]

    X_points, O_points = xoc(
        unclassified_fixed_points=fixed_points,
        profile=giorgos_efield_profile,
        to_P_thetas=False,
    )

    assert len(X_points) == len(O_points)
