import pytest
import numpy as np
import gcmotion as gcm


@pytest.fixture(scope="function")
def events(simple_particle, request, terminal):
    r"""All availiable events with simple_particle's parameters."""
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


@pytest.fixture(scope="session")
def long_init(Q):
    r"""Initial conditions object with very long integration time."""
    return gcm.InitialConditions(
        species="p",
        muB=Q(0.5, "keV"),
        theta0=1,
        zeta0=0,
        psi0=Q(0.8, "psi_wall"),
        Pzeta0=Q(0.02, "NUCanonical_momentum"),
        t_eval=Q(np.linspace(0, 1e-4, 100000), "seconds"),
    )


@pytest.fixture(scope="function")
def long_particle(simple_tokamak, long_init, Q):
    """Simple Particle object (B=LAR, q=Unity, E=Nofield) with very long
    integration time."""
    particle = gcm.Particle(tokamak=simple_tokamak, init=long_init)
    yield particle


@pytest.fixture(scope="session")
def simple_particle_single_period(simple_tokamak, simple_init, Q):
    """Simple Particle object (B=LAR, q=Unity, E=Nofield) with very long
    integration time."""
    particle = gcm.Particle(tokamak=simple_tokamak, init=simple_init)
    particle.run(method="NPeriods", stop_after=1)
    return particle
