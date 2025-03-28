import gcmotion as gcm
import numpy as np


def test_initial_conditions_muB_keV(simple_tokamak, Q):
    init = gcm.InitialConditions(
        species=Q.args["species"],
        muB=Q(0.5, "keV"),
        Pzeta0=Q(-0.025, "NUCanonical_momentum"),
        theta0=np.pi,
        zeta0=0,
        psi0=Q(0.132, "Magnetic_flux"),
        t_eval=Q(np.linspace(0, 1e-3, 100000), "seconds"),
    )
    # Make sure these work in both cases since they change after
    # _calculate_full_set() is called
    init.__repr__()
    init.__str__()

    init._calculate_full_set(simple_tokamak)

    # Make sure these work in both cases
    init.__repr__()
    init.__str__()

    assert (
        str(init.muB.dimensionality) == "[mass] * [length] ** 2 / [time] ** 2"
    )
    assert str(init.mu.dimensionality) == "[current] * [length] ** 2"


def test_initial_conditions_muB_magnetic_moment(simple_tokamak, Q):
    init = gcm.InitialConditions(
        species=Q.args["species"],
        muB=Q(1e-6, "NUMagnetic_moment"),
        Pzeta0=Q(-0.025, "NUCanonical_momentum"),
        theta0=np.pi,
        zeta0=0,
        psi0=Q(0.132, "Magnetic_flux"),
        t_eval=Q(np.linspace(0, 1e-3, 100000), "seconds"),
    )
    # Make sure these work in both cases since they change after
    # _calculate_full_set() is called
    init.__repr__()
    init.__str__()

    init._calculate_full_set(simple_tokamak)

    # Make sure these work in both cases
    init.__repr__()
    init.__str__()

    assert (
        str(init.muB.dimensionality) == "[mass] * [length] ** 2 / [time] ** 2"
    )
    assert str(init.mu.dimensionality) == "[current] * [length] ** 2"


def test_initial_conditions_no_t_eval(simple_tokamak, Q):
    init = gcm.InitialConditions(
        species=Q.args["species"],
        muB=Q(1e-6, "NUMagnetic_moment"),
        Pzeta0=Q(-0.025, "NUCanonical_momentum"),
        theta0=np.pi,
        zeta0=0,
        psi0=Q(0.132, "Magnetic_flux"),
    )
    # Make sure these work in both cases since they change after
    # _calculate_full_set() is called
    init.__repr__()
    init.__str__()

    init._calculate_full_set(simple_tokamak)

    # Make sure these work in both cases
    init.__repr__()
    init.__str__()

    assert init.t_eval is None
