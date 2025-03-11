r"""
Simple script that calculates the RW,LW,MA parabolas diagram (constant :math:`\mu` slices
in the :math:`\theta`,:math:`P_{\zeta}`,E COM space).
"""

import numpy as np
from gcmotion.utils.logger_setup import logger
from collections import deque
from time import time

from gcmotion.entities.profile import Profile
from gcmotion.entities.tokamak import Tokamak
from gcmotion.tokamak.efield import Nofield

from gcmotion.configuration.scripts_configuration import ParabolasConfig

from gcmotion.scripts.fixed_points_bif.bifurcation import bifurcation


def calc_parabolas_tpb(
    profile: Profile,
    calc_TPB: bool = False,
    **kwargs,
) -> dict:
    r"""

    In this script the values of the parabolas that correspond to the right and left
    boundary and to the magnetic axis, are calculated, along with the trapped passing
    boundary. CUTION: this script provides correct values only if there is no
    electric field.

    Parameters
    ----------
    profile : Profile
        Profile object that contains Tokamak information like bfield, mu,
        useful psi values.
    Pzetalim : tuple,list, optional
        The Pzeta limits within which the RW, LW, MA parabolas' values are to
        be calculated. CAUTION: the limits must be normalized to psip_wallNU.
        Defaults to (-1.5,1).
    Pzeta_density : int, optional
        The density of :math:`P_{\zeta}` points that will be used to calculate
        the RW, LW, MA parabolas' values. Defaults to 1000.
    calc_TPB : bool, optional
        Boolean that determines weather the values for the trapped passing
        boundary are to be calculated. Defaults to ``False``.
    TPB_density : int, optional
        The points density of :math:`P_{\zeta}`s that will be used to calculate
        the trapped passing boundary. Defaults to 100.

    Returns
    -------
    dict
    Dict that contains the the x values x and x_TPB that were used to calculate the parabolas
    values and the trapped passing boundary values respectively, y_R,y_L,Y_MA the parabolas values
    for the right and left wall and magnetic axis parabolas, TPB_O, TPB_X the lower and upper
    branch(es) of the trapped passing boundary curve respectively, v_R, v_L, v_MA the location (points)
    of the minima (vertex) of each parabola.

    """

    # Unpack parameters
    config = ParabolasConfig()
    for key, value in kwargs.items():
        setattr(config, key, value)

    # First of all check if there is an electric field
    if not isinstance(profile.efield, Nofield):
        print("\n\nWARNING: Parabolas do not work with an electric field...\n\n")

    # -------------------------------------------------------------------------------

    # Unpack magnitudes
    bfield = profile.bfield
    psip_wallNU = profile.psip_wallNU.m
    psi_wallNU = profile.psi_wallNU.m
    muNU = profile.muNU.m

    # Unpack Pzeta limits
    PzetaminNU, PzetamaxNU = config.Pzetalim
    PzetasNU = np.linspace(PzetaminNU, PzetamaxNU, config.Pzeta_density)

    logger.info(f"Unpacked parabolas Pzetalim={config.Pzetalim}")

    # Get intermediate values
    try:
        BminNU = bfield.Bmin.to("NUTesla").m[0]  # List if numerical
    except:
        BminNU = bfield.Bmin.to("NUTesla").m

    BmaxNU = bfield.Bmax.to("NUTesla").m
    B0NU = bfield.B0.to("NUTesla").m
    muB0NU = muNU * B0NU

    logger.info(f"Upacked parabolas bfield values with Bmin={BminNU},Bmax={BmaxNU},B0={B0NU}")

    # Because Pzetas given in Pzetalim are already divided (normalized) by psip_wall
    x = PzetasNU

    # Currents are poloidally symmetrical --> independent of theta. Calculate g.
    _, _, g_psipwNU = bfield.bigNU(psip_wallNU, 0)
    _, _, g0NU = bfield.bigNU(0, 0)

    logger.info(f"Upacked parabolas g values with g(psip_wall)={g_psipwNU}, g(0)={g0NU}")

    # -------------------------------------------------------------------------------

    # Calculate the quantity y=E/(mu*B0) to be plotted on the y axis
    # as a function of the quantity x=Pzeta/psi_pwall for x axis
    y_R = 1 / (2 * muB0NU) * (psip_wallNU * BminNU / g_psipwNU) ** 2 * (x + 1) ** 2 + BminNU / B0NU

    y_L = 1 / (2 * muB0NU) * (psip_wallNU * BmaxNU / g_psipwNU) ** 2 * (x + 1) ** 2 + BmaxNU / B0NU

    y_MA = 1 / (2 * muB0NU) * (psip_wallNU * B0NU / g0NU) ** 2 * (x) ** 2 + 1

    logger.info("Calculated parabolas y(x) values where y=E/muB0 and x=Pz/psip_wall")

    # Calculate the vertex points of the three parabolas (will be useful)
    v_R = [x[np.argmin(y_R)], y_R[np.argmin(y_R)]]
    v_L = [x[np.argmin(y_L)], y_L[np.argmin(y_L)]]
    v_MA = [x[np.argmin(y_MA)], y_MA[np.argmin(y_MA)]]

    # -------------------------------------------------------------------------------

    # Calculate the  trapped-passing boundary
    TPB_O = np.array([])
    TPB_X = np.array([])
    x_TPB = np.linspace(v_R[0], v_MA[0], config.Pzeta_density)

    if calc_TPB and not bfield.plain_name == "LAR":
        # If the equilibrium is not LAR we need to constrauct the energy bifurcation
        # diagram with respect to Pzeta, which is the trapped passing boundary.

        # Create profiles that will be passed into bifurcation. CAUTION: the Pzetas
        # are in NUcanmom and not divided by psip_wallNU
        Pzetamin, Pzetamax = min(x_TPB), max(x_TPB)  # PzetaNU/psip_wallNU
        Pzetamin *= psip_wallNU  # Pzeta in NUcanmom
        Pzetamax *= psip_wallNU  # Pzeta in NUcanmom

        PzetasNU = np.linspace(Pzetamin, Pzetamax, config.Pzeta_density)

        start = time()
        # Energies are returned in NUJoule
        bifurcation_output = bifurcation(
            profile=profile,
            COM_values=PzetasNU,
            calc_energies=True,
            energy_units="NUJoule",
            which_COM="Pzeta",
            **kwargs,
        )

        print(f"BIFURCATION RUN IN {(time() - start)/60:.1f} mins")

        # Unpack bifurcation output
        X_energies = bifurcation_output["X_energies"]
        O_energies = bifurcation_output["O_energies"]

        # We do not yet divide by B0muNU yet as the energies will be needed later in NUJoule
        # Bifurcation returns deque() objects. We cast into lists.
        TPB_O = list(O_energies)
        TPB_X = list(X_energies)

        logger.info(
            f"Ran bifurcation for parabolas TPB for {config.TPB_density} profiles. Calculated energies in 'NUJoule'"
        )

    else:
        # 0 <psi<psi_wall and at trapped - passing boundary E_O/(μΒ0)=B(θ=0,psi)/B0
        #                        and                        E_X/(μΒ0)=B(θ=π,psi)/B0
        psis = np.linspace(psi_wallNU, 0, config.Pzeta_density)
        TPB_O, _, _ = bfield.bigNU(psis, 0)  # E_O(x)/μ
        TPB_X, _, _ = bfield.bigNU(psis, np.pi)  # E_X(x)/μ

        TPB_O /= B0NU  # E_O/(μΒ0)
        TPB_X /= B0NU  # E_X/(μΒ0)

        logger.info(f"Calculated LAR TPB boundary with x axis limits: ({min(x_TPB)},{max(x_TPB)})")

    return {
        "x": x,
        "x_TPB": x_TPB,
        "y_R": y_R,
        "y_L": y_L,
        "y_MA": y_MA,
        "TPB_O": TPB_O,
        "TPB_X": TPB_X,
        "v_R": v_R,
        "v_L": v_L,
        "v_MA": v_MA,
    }
