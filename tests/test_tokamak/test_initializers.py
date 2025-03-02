import pytest
import gcmotion as gcm

initializers = [
    gcm.SmartPositiveInit,
    gcm.SmartNegativeInit,
    gcm.SmartNegative2Init,
    gcm.DTTPositiveInit,
    gcm.DTTNegativeInit,
]


@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
def test_initializers_initialization():
    r"""Tests that all initializers are initialized."""
    for init in initializers:
        init = init(species="T")
        init.QuantityConstructor()
        assert hasattr(init, "R")
        assert hasattr(init, "B0")
        assert hasattr(init, "psi_wallNU")
        assert hasattr(init, "a")


def test_initializer_missing_dataset():
    with pytest.raises(FileNotFoundError):
        gcm.tokamak.reconstructed.initializers._NumericalInitializer(
            species="p", filename="not_a_file.nc"
        )
