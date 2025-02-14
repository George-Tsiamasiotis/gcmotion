r"""
Simple script that calculates the RW,LW,MA parabolas diagram (constant :math:`\mu` slices
in the :math:`\theta`,:math:`P_{\zeta}`,E COM space).
"""

import numpy as np
from gcmotion.utils.logger_setup import logger
from gcmotion.entities.profile import Profile
from gcmotion.tokamak.efield import Nofield

from gcmotion.configuration.parabolas_parameters import ParabolasConfig


def calc_parabolas(profile: Profile, **kwargs):
    r"""

    In this script the values of the parabolas that correspond to the right and left
    boundary and to the magnetic axis, are calculated, along with the trapped passing
    boundary for LAR. CUTION: this script provides correct values only if there is no
    electric field.

    Parameters
    ----------
    profile : Profile
        Profile object that contains Tokamak information like bfield, mu,
        useful psi values.

    Other Parameters
    ----------
    Pzetalim : tuple,list, optional
        The Pzeta limits within which the RW, LW, MA parabolas' values are to
        be calculated. CAUTION: the limits must be normalized to psip_wallNU.
        Defaults to (-1.5,1).
    Pzeta_density : int, optional
        The density of Pzeta points that will be used to calculate
        the RW, LW, MA parabolas' values. Defaults to 1000.


    Returns
    -------
    x, x_LAR, y_R, y_L, y_MA, LAR_tpb_O, LAR_tpb_X : tuple
    Tuple that contains the the x values x and x_LAR that were used to calculate the parabolas
    values and the trapped passing LAR boundary values respectively, y_R,y_L,Y_MA the parabolas values
    for th right and left wall and magnetic axis parabolas, LAR_tpb_O, LAR_tpb_X the lower and upper
    branch of the trapped passing LAR boundary curve respetively.

    """

    # First of all check if there is an electric field
    if not isinstance(profile.efield, Nofield):
        print("\n\nWARNING: Parabolas do not work with an electric field...\n\n")

    # Unpack Parameters
    config = ParabolasConfig()
    for key, value in kwargs.items():
        setattr(config, key, value)

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

    # Because Pzetas given in Pzetalim are already divided (normalized) by pasip_wall
    x = PzetasNU

    # Currents are poloidally symmetrical --> independent of theta. Calculate g.
    _, _, g_psipwNU = bfield.bigNU(psip_wallNU, 0)
    _, _, g0NU = bfield.bigNU(0, 0)

    logger.info(f"Upacked parabolas g values with g(psip_wall)={g_psipwNU},g(0)={g0NU}")

    # Calculate the quantity y=E/(mu*B0) to be plotted on the y axis
    # as a function of the quantity x=Pzeta/psi_pwall for x axis
    y_R = 1 / (2 * muB0NU) * (psip_wallNU * BminNU / g_psipwNU) ** 2 * (x + 1) ** 2 + BminNU / B0NU

    y_L = 1 / (2 * muB0NU) * (psip_wallNU * BmaxNU / g_psipwNU) ** 2 * (x + 1) ** 2 + BmaxNU / B0NU

    y_MA = 1 / (2 * muB0NU) * (psip_wallNU * B0NU / g0NU) ** 2 * (x) ** 2 + 1

    logger.info("Calculated parabolas y(x) values where y=E/muB0 and x=Pz/psip_wall")

    # Calculate the LAR trapped-passing boundary
    LAR_tpb_O = np.array([])
    LAR_tpb_X = np.array([])
    x_LAR = np.linspace(PzetasNU[np.argmin(y_L)], PzetasNU[np.argmin(y_MA)], config.Pzeta_density)

    if not bfield.plain_name == "LAR":
        print(
            "\nWARNING: Can not calculate LAR trapped-passing boundary analytically,"
            "for non LAR equilibrium. Returning empty LAR_tpb...\n"
        )
    else:
        # 0 <psi<psi_wall and at trapped - passing boundary E_O/μΒ0=B(θ=0,psi)/B0
        #                        and                        E_X/μΒ0=B(θ=π,psi)/B0
        # Also ψp=ψ and Pz=-ψp=-ψ --> ψ=-ψpw*x where x=Pz/ψpw.
        # So E_O/μΒ0=B(θ=0,-ψpw*x)/B0 and E_X/μΒ0=B(θ=π,-ψpw*x)/B0

        LAR_tpb_O, _, _ = bfield.bigNU(-x_LAR * psi_wallNU, 0)  # E_O(x)
        LAR_tpb_X, _, _ = bfield.bigNU(-x_LAR * psi_wallNU, np.pi)  # E_X(x)

        LAR_tpb_O /= B0NU  # E_O/μΒ0
        LAR_tpb_X /= B0NU  # E_X/μΒ0

        logger.info(f"Calculated tpb LAR boundary with x axis limits: ({min(x_LAR)},{max(x_LAR)})")

    return x, x_LAR, y_R, y_L, y_MA, LAR_tpb_O, LAR_tpb_X
