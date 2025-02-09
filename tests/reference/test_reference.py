import doctest
import pytest

V = False  # Verbosity


def test_quantity_reference(reference_path):
    doctest_results = doctest.testfile(
        filename=str(reference_path / "quantity.rst"), verbose=V
    )
    assert doctest_results[0] == 0


@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
def test_qfactor_reference(reference_path):
    doctest_results = doctest.testfile(
        filename=str(reference_path / "qfactor.rst"), verbose=V
    )
    assert doctest_results[0] == 0


@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
def test_bfield_reference(reference_path):
    doctest_results = doctest.testfile(
        filename=str(reference_path / "bfield.rst"), verbose=V
    )
    assert doctest_results[0] == 0


def test_efield_reference(reference_path):
    doctest_results = doctest.testfile(
        filename=str(reference_path / "efield.rst"), verbose=V
    )
    assert doctest_results[0] == 0


@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
def test_initializers_reference(reference_path):
    doctest_results = doctest.testfile(
        filename=str(reference_path / "initializers.rst"), verbose=V
    )
    assert doctest_results[0] == 0
