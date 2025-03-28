from math import isclose


def test_RK45(simple_particle):
    simple_particle.run(method="RK45")

    assert simple_particle.method == "RK45"
    assert isclose(simple_particle.orbit_percentage, 100)
