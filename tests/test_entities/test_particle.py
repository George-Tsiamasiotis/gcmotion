import pytest


def test_particle_run_no_args(simple_particle):
    simple_particle.run()
    assert simple_particle.method == "RK45"


@pytest.mark.parametrize("which", ("init", ""))
@pytest.mark.parametrize("everything", ("SI", "NU"))
def test_particle_print_quantities(simple_particle, which, everything):
    simple_particle.quantities(which=which, everything=everything)


def test_particle_repr_str(simple_particle):
    simple_particle.__repr__()
    simple_particle.__str__()
