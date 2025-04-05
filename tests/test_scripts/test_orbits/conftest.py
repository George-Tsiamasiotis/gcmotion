import pytest
import numpy as np
import gcmotion as gcm


@pytest.fixture(scope="session")
def events_particle(simple_tokamak, Q):
    r"""Particle used for events testing."""
    events_init = gcm.InitialConditions(
        species="p",
        muB=Q(0.5, "keV"),
        theta0=1,
        zeta0=0,
        psi0=Q(0.5, "psi_wall"),
        Pzeta0=Q(0.02, "NUCanonical_momentum"),
        t_eval=Q(np.linspace(0, 1e-5, 10000), "seconds"),
    )
    particle = gcm.Particle(tokamak=simple_tokamak, init=events_init)
    return particle


@pytest.fixture(scope="function")
def events(simple_particle, request, terminal):
    r"""Requested events."""
    theta = simple_particle.theta0
    psi = simple_particle.psi0NU
    zeta = simple_particle.zeta0
    rho = simple_particle.rho0

    if request.param == "when_theta":
        return gcm.events.when_theta(root=theta, terminal=terminal)
    if request.param == "when_psi":
        return gcm.events.when_psi(root=psi, terminal=terminal)
    if request.param == "when_zeta":
        return gcm.events.when_zeta(root=zeta, terminal=terminal)
    if request.param == "when_rho":
        return gcm.events.when_rho(root=rho, terminal=terminal)


# =============================================================================


def nperiods_particle_trapped(simple_tokamak, Q):
    """Trapped particle to be used with NPeriods Solver testing."""
    init = gcm.InitialConditions(
        species="p",
        muB=Q(50, "keV"),
        theta0=1,
        zeta0=0,
        psi0=Q(0.53, "psi_wall"),
        Pzeta0=Q(-0.02, "NUCanonical_momentum"),
    )
    particle = gcm.Particle(tokamak=simple_tokamak, init=init)
    return particle


def nperiods_particle_passing(simple_tokamak, Q):
    """Passing particle to be used with NPeriods Solver testing."""
    init = gcm.InitialConditions(
        species="p",
        muB=Q(50, "keV"),
        theta0=1,
        zeta0=0,
        psi0=Q(0.7, "psi_wall"),
        Pzeta0=Q(-0.02, "NUCanonical_momentum"),
    )
    particle = gcm.Particle(tokamak=simple_tokamak, init=init)
    return particle


@pytest.fixture(scope="function")
def nperiods_particle(simple_tokamak, Q, request):
    r"""Requested trapped and passing NPeriod Solver particles."""
    if request.param == "trapped":
        return nperiods_particle_trapped(simple_tokamak, Q)
    if request.param == "passing":
        return nperiods_particle_passing(simple_tokamak, Q)
