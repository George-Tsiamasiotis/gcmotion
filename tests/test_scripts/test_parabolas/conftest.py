import pytest
from gcmotion.scripts.parabolas import calc_parabolas_tpb


@pytest.fixture(scope="module")
def parabolas_output(simple_profile):
    r"""Calculates tha values of the parabolas and the TP-boundary
    for a simple profile."""
    parabolas_output = calc_parabolas_tpb(
        profile=simple_profile, calc_TPB=True
    )

    return parabolas_output
