import pytest
import doctest
from gcmotion.scripts import events
from gcmotion.entities import (
    tokamak,
    initial_conditions,
    profile,
    particle,
)
from gcmotion.scripts.frequency_analysis import frequency_analysis

V = True  # Verbosity


def test_tokamak_docstring():
    doctest_results = doctest.testmod(m=tokamak, verbose=V)
    assert doctest_results.failed == 0


def test_initial_conditions_docstring():
    doctest_results = doctest.testmod(m=initial_conditions, verbose=V)
    assert doctest_results.failed == 0


def test_profile_docstring():
    doctest_results = doctest.testmod(m=profile, verbose=V)
    assert doctest_results.failed == 0


def test_particle_docstring():
    with pytest.warns(UserWarning):
        doctest_results = doctest.testmod(m=particle, verbose=V)
    assert doctest_results.failed == 0


def test_events_docstring():
    doctest_results = doctest.testmod(m=events, verbose=V)
    assert doctest_results.failed == 0


def test_frequency_analysis_docstring():
    doctest_results = doctest.testmod(m=frequency_analysis, verbose=V)
    assert doctest_results.failed == 0
