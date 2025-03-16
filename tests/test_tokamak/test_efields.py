r"""
Tests that all analytical efields' methods work, and querry methods return the
desired type.

Tests
-----
1. Analytical/Numerical flags
2. __repr__() functionality
3. bigNU return types
4. solverbNU return types

"""

import pytest
import numpy as np


@pytest.mark.parametrize(
    "efields",
    [
        "nofield",
        "cosine_potential",
        "radial",
    ],
    indirect=True,
)
class TestAnalytical:

    psi_float = 0.02
    psi_array = np.linspace(1e-5, 0.05, 10)
    theta_float = np.pi / 2
    theta_array = np.linspace(-np.pi / 2, np.pi / 3, 10)

    def test_flags(self, efields):
        assert efields.is_analytical
        assert not efields.is_numerical

    def test_repr_functionality(self, efields):
        efields.__repr__()

    def test_solverPhiderNU_return_type(self, efields):
        r"""Must be able to return ONLY floats."""

        der1_float, der2_float = efields.solverPhiderNU(
            self.psi_float, self.theta_float
        )

        assert isinstance(der1_float, (float, int))
        assert isinstance(der2_float, (float, int))

    def test_PhiNU_return_type(self, efields):
        r"""Must be able to return both floats and numpy arrays."""

        phi_float = efields.PhiNU(self.psi_float, self.theta_float)
        phi_array = efields.PhiNU(self.psi_array, self.theta_array)

        assert isinstance(phi_float, (float, int))
        assert isinstance(phi_array, np.ndarray)

    def test_Er(self, efields):
        r"""Must return np.ndarray (or np.float64), since its only used for
        plotting.
        """
        Er = efields.Er(self.psi_array, self.theta_array)
        assert isinstance(Er, np.ndarray)
