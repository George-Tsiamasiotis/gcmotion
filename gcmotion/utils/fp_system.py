r"""
Function that calculates the time derivatives of the :math:`\theta` and :math:`\psi`
variables, :math:`\dot{\theta}` and :math:`\dot{\psi}` respectively, according to White's equations.
"""

from gcmotion.entities.profile import Profile


# System of equations to be solved
def system(theta: float, psi: float, profile: Profile, method: str):
    r"""
    Function that calculates the time derivatives of the :math:`\theta` and :math:`\psi`
    variables, based on White's equations. It will be used in the :py:func:`single_fixed_point`
    and :py:func:`fp_ic_scan` methods, to evaluate for which :math:`\theta` and :math:`\psi`
    values the derivatives become zero.

        Parameters
        ----------
        theta : float
            Value of the :math:`\theta` for which the time derivatives will be calculated.
        psi : float
            Value of the :math:`\psi` for which the time derivatives will be calculated.
        profile : Profile
            Profile object that contains Tokamak and Particle information.
        method : str
            String that indicates which method will be used to find the systems fixed
            points in :py:func:`single_fixed_point`. Can either be "fsolve" (deterministic)
            or "differential evolution" (stochastic).



        Returns
        -------
        The time derivatives of the :math:`\theta` and :math:`\psi`
        variables or the sum of the time derivatives' squares (depends on the
        selected method)

    """

    # Tokamak profile
    qfactor = profile.qfactor
    bfield = profile.bfield
    efield = profile.efield

    # Define quantites for the solver for clarity
    solverqNU = qfactor.solverqNU
    psipNU = qfactor.psipNU
    solverbNU = bfield.solverbNU
    solverPhiderNU = efield.solverPhiderNU

    # Parameters ()
    mu = profile.muNU.m
    Pzeta0 = profile.PzetaNU.m
    Pzeta = Pzeta0

    psi = max(psi, 1e-12)

    # Object methods calls
    q = solverqNU(psi)
    psi_p = psipNU(psi)
    b, b_der, currents, currents_der = solverbNU(psi, theta)
    phi_der_psi, phi_der_theta = solverPhiderNU(psi, theta)

    # Unpack
    b_der_psi, b_der_theta = b_der
    i, g = currents
    i_der, g_der = currents_der
    # Multiply current derivatives by q to get their derivatives with
    # respect to psi instead of psip
    i_der, g_der = q * i_der, q * g_der

    # Intermediate values
    rho = (Pzeta + psi_p) / g
    par = mu + rho**2 * b
    bracket1 = -par * b_der_psi + phi_der_psi
    bracket2 = par * b_der_theta + phi_der_theta
    D = g * q + i + rho * (g * i_der - i * g_der)

    # Canonical Equations
    theta_dot = (1 - rho * g_der) / D * rho * b**2 + q * g / D * bracket1
    psi_dot = -q * g / D * bracket2

    if method == "differential evolution":
        return theta_dot**2 + (70 * psi_dot) ** 2

    elif method == "fsolve":
        return [theta_dot, psi_dot]
