r"""
Tests that all bfields' methods work, and querry methods return the desired
type.

Tests
-----
1. Analytical/Numerical flags
2. __repr__() functionality
3. bigNU return types
4. solverbNU return types

"""

import pint
import pytest
import numpy as np
import gcmotion as gcm


@pytest.mark.parametrize(
    "bfields",
    [
        "lar",
    ],
    indirect=True,
)
class TestAnalytical:

    psi_float = 0.02
    psi_array = np.linspace(1e-5, 0.05, 10)
    theta_float = np.pi / 2
    theta_array = np.linspace(-np.pi / 2, np.pi / 3, 10)

    def test_flags(self, bfields):
        assert bfields.is_analytical
        assert not bfields.is_numerical

    def test_repr_functionality(self, bfields):
        bfields.__repr__()

    def test_bigNU_return_type(self, bfields):
        r"""Must be able to return both floats and numpy arrays."""

        b_float, i_float, g_float = bfields.bigNU(
            self.psi_float, self.theta_float
        )
        b_array, i_array, g_array = bfields.bigNU(
            self.psi_array, self.theta_array
        )

        assert isinstance(b_float, (float, int))
        assert isinstance(i_float, (float, int))
        assert isinstance(g_float, (float, int))
        assert isinstance(b_array, np.ndarray)
        assert isinstance(i_array, np.ndarray)
        assert isinstance(g_array, np.ndarray)

    def test_solverbNU_return_type(self, bfields):
        r"""Must be able to return ONLY floats."""

        b, b_der, currents, currents_der = bfields.solverbNU(
            self.psi_float, self.theta_float
        )
        for values in (b, *b_der, *currents, *currents_der):
            assert isinstance(values, (float, int))

    def test_bmin_bmax(self, bfields):
        assert isinstance(bfields.Bmin, pint.registry.Quantity)
        assert isinstance(bfields.Bmax, pint.registry.Quantity)

        assert bfields.Bmin.ndim == 0
        assert bfields.Bmax.ndim == 0


"""
Return if the fixture returns None, which means the corresponding dataset does
not exist.
"""


@pytest.mark.parametrize(
    "bfields",
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
    theta_float = np.pi / 2
    theta_array = np.linspace(-np.pi / 2, np.pi / 3, 10)

    def test_flags(self, bfields):
        if bfields is None:
            return

        assert not bfields.is_analytical
        assert bfields.is_numerical

    def test_repr_functionality(self, bfields):
        if bfields is None:
            return

        bfields.__repr__()

    def test_bigNU_return_type(self, bfields):
        r"""Must be able to return both floats and numpy arrays."""
        if bfields is None:
            return

        b_float, i_float, g_float = bfields.bigNU(
            self.psi_float, self.theta_float
        )
        b_array, i_array, g_array = bfields.bigNU(
            self.psi_array, self.theta_array
        )

        assert isinstance(b_float, (float, int))
        assert isinstance(i_float, (float, int))
        assert isinstance(g_float, (float, int))
        assert isinstance(b_array, np.ndarray)
        assert isinstance(i_array, np.ndarray)
        assert isinstance(g_array, np.ndarray)

    def test_solverbNU_return_type(self, bfields):
        r"""Must be able to return ONLY floats."""
        if bfields is None:
            return

        b, b_der, currents, currents_der = bfields.solverbNU(
            self.psi_float, self.theta_float
        )
        for values in (b, *b_der, *currents, *currents_der):
            assert isinstance(values, (float, int))

    def test_bmin_bmax(self, bfields):
        if bfields is None:
            return

        assert isinstance(bfields.Bmin, pint.registry.Quantity)
        assert isinstance(bfields.Bmax, pint.registry.Quantity)

        assert bfields.Bmin.ndim == 0
        assert bfields.Bmax.ndim == 0


@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
def test_numerical_missing_dataset():
    with pytest.raises(FileNotFoundError):
        gcm.bfield.NumericalMagneticField(filename="not_a_file.nc")
