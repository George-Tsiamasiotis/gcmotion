import pytest
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
