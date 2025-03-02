r"""
Tests that all qfactors' methods work, and querry methods return the desired
type.

Tests
-----
1. Analytical/Numerical flags
2. __repr__() functionality
3. solverqNU return types
4. psipNU return types

"""

import pytest
import numpy as np
import gcmotion as gcm


@pytest.mark.parametrize(
    "qfactors",
    [
        "unity",
        "parabolic",
        "hypergeometric",
        "precomputed hypergeometric",
    ],
    indirect=True,
)
class TestAnalytical:

    psi_float = 0.02
    psi_array = np.linspace(1e-5, 0.05, 10)

    def test_flags(self, qfactors):
        assert qfactors.is_analytical
        assert not qfactors.is_numerical

    def test_repr_functionality(self, qfactors):
        qfactors.__repr__()

    def test_solverqNU_return_type(self, qfactors):
        r"""Must be able to return both floats and numpy arrays."""

        q_float = qfactors.solverqNU(self.psi_float)
        q_array = qfactors.solverqNU(self.psi_array)

        assert isinstance(q_float, (float, int))
        assert isinstance(q_array, np.ndarray)

    def test_psipNU_return_type(self, qfactors):
        r"""Must be able to return both floats and numpy arrays."""

        q_float = qfactors.psipNU(self.psi_float)
        q_array = qfactors.psipNU(self.psi_array)

        assert isinstance(q_float, (float, int))
        assert isinstance(q_array, np.ndarray)


"""
Return if the fixture returns None, which means the corresponding dataset does
not exist.
"""


@pytest.mark.parametrize(
    "qfactors",
    [
        "smart_pt",
        "smart_nt",
        "smart_nt2",
        "dtt_pt",
        "dtt_nt",
    ],
    indirect=True,
)
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
class TestNumerical:

    psi_float = 0.02
    psi_array = np.linspace(1e-5, 0.05, 10)

    def test_flags(self, qfactors):
        if qfactors is None:
            return

        assert not qfactors.is_analytical
        assert qfactors.is_numerical

    def test_repr_functionality(self, qfactors):
        if qfactors is None:
            return

        qfactors.__repr__()

    def test_solverqNU_return_type(self, qfactors):
        r"""Must be able to return both floats and numpy arrays."""
        if qfactors is None:
            return

        q_float = qfactors.solverqNU(self.psi_float)
        q_array = qfactors.solverqNU(self.psi_array)

        assert isinstance(q_float, (float, int))
        assert isinstance(q_array, np.ndarray)

    def test_psipNU_return_type(self, qfactors):
        r"""Must be able to return both floats and numpy arrays."""
        if qfactors is None:
            return

        q_float = qfactors.psipNU(self.psi_float)
        q_array = qfactors.psipNU(self.psi_array)

        assert isinstance(q_float, (float, int))
        assert isinstance(q_array, np.ndarray)


@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
def test_numerical_missing_dataset():
    with pytest.raises(FileNotFoundError):
        gcm.qfactor.NumericalQFactor(filename="not_a_file.nc")
