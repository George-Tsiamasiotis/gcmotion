r"""
===================
Initializers Module
===================

"""

import os
import xarray as xr

from gcmotion.utils.quantity_constructor import QuantityConstructor


class _NumericalInitializer:
    r"""Imports the necessary constants (R, B0, :math:`\psi_{wall}`) from the
    dataset to initialize the Quantity Constructor.

    The particle's species must be defined by the user, while the rest of the
    Quantities are already stored in the dataset.

    Parameters
    ----------
    species : {'p', 'e', 'D', 'T', 'He3', 'He4'}
        The particle's species. This field is case-insensitive.

    """

    def __init__(self, species: str, filename: str):

        # Open the dataset
        parent = os.path.dirname(__file__)
        path = os.path.join(parent, filename)
        try:
            ds = xr.open_dataset(path)
        except FileNotFoundError:
            raise FileNotFoundError(f"No file found at '{path}'")

        # Get numerical values from dataset
        B0 = float(ds.Baxis.data)  # Tesla
        R = ds.raxis.data  # meters
        psi_wallNU = float(ds.psi[-1].data)  # NUMagnetic_flux

        # Create Quantity Constructor
        self.Q = QuantityConstructor(
            R=R,
            B0=B0,
            _psi_wallNU=psi_wallNU,
            species=species,
        )

        # Create Quantities
        self.R = self.Q(R, "meters")
        self.B0 = self.Q(B0, "Tesla")
        self.psi_wallNU = self.Q(psi_wallNU, "NUMagnetic_flux")
        psi_wall = self.psi_wallNU.to("Tesla * meters^2")
        self.a = (2 * psi_wall / self.B0) ** (1 / 2)

    def QuantityConstructor(self):
        r"""Return the Quantity Constructor Q, defined with the datasets
        parameters."""
        return self.Q


class SmartPositiveInit(_NumericalInitializer):
    r"""Imports the necessary constants (R, B0, :math:`\psi_{wall}`) from the
    **Smart - Positive** dataset to initialize the Quantity Constructor.

    The particle's species must be defined by the user, while the rest of the
    Quantities are already stored in the dataset.

    Dataset location should be at
    gcmotion/tokamak/reconstructed/smart_positive.nc

    Parameters
    ----------
    species : {'p', 'e', 'D', 'T', 'He3', 'He4'}
        The particle's species. This field is case-insensitive.

    """

    def __init__(self, species: str):
        filename = "smart_positive.nc"
        super().__init__(filename=filename, species=species)


class SmartNegativeInit(_NumericalInitializer):
    r"""Imports the necessary constants (R, B0, :math:`\psi_{wall}`) from the
    **Smart - Negative** dataset to initialize the Quantity Constructor.

    The particle's species must be defined by the user, while the rest of the
    Quantities are already stored in the dataset.

    Dataset location should be at
    gcmotion/tokamak/reconstructed/smart_negative.nc

    Parameters
    ----------
    species : {'p', 'e', 'D', 'T', 'He3', 'He4'}
        The particle's species. This field is case-insensitive.

    """

    def __init__(self, species: str):
        filename = "smart_negative.nc"
        super().__init__(filename=filename, species=species)


class DivertorNegativeInit(_NumericalInitializer):
    r"""Imports the necessary constants (R, B0, :math:`\psi_{wall}`) from the
    **Divertor - Negative** dataset to initialize the Quantity Constructor.

    The particle's species must be defined by the user, while the rest of the
    Quantities are already stored in the dataset.

    Dataset location should be at
    gcmotion/tokamak/reconstructed/divertor_negative.nc

    Parameters
    ----------
    species : {'p', 'e', 'D', 'T', 'He3', 'He4'}
        The particle's species. This field is case-insensitive.

    """

    def __init__(self, species: str):
        filename = "divertor_negative.nc"
        super().__init__(filename=filename, species=species)
