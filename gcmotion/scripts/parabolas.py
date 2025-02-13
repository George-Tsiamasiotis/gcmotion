import numpy as np
from gcmotion.utils.logger_setup import logger
from gcmotion.entities.profile import Profile
from gcmotion.tokamak.efield import Nofield

from gcmotion.configuration.parabolas_parameters import ParabolasConfig


def calc_parabolas(profile: Profile, **kwargs):

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

    # Get intermediate values
    BminNU = bfield.Bmin.to("NUTesla").m
    BmaxNU = bfield.Bmax.to("NUTesla").m
    B0NU = bfield.B0.to("NUTesla").m
    muB0NU = muNU * B0NU

    # Becaues Pzetas given in Pzetalim are already divided (normalized) by pasip_wall
    x = PzetasNU

    # Currents are poilooidally symmetrical --> independent of theta
    _, _, g_psipwNU = bfield.bigNU(psip_wallNU, 0)
    _, _, g0NU = bfield.bigNU(0, 0)

    # Calculate the quantity E/(mu*B0) to be plotted on the y axis
    # as a function of the quantity Pzeta/psi_pwall for x axis
    y_R = 1 / (2 * muB0NU) * (psip_wallNU * BminNU / g_psipwNU) ** 2 * (x + 1) ** 2 + BminNU / B0NU

    y_L = 1 / (2 * muB0NU) * (psip_wallNU * BmaxNU / g_psipwNU) ** 2 * (x + 1) ** 2 + BmaxNU / B0NU

    y_MA = 1 / (2 * muB0NU) * (psip_wallNU * B0NU / g0NU) ** 2 * (x) ** 2 + 1

    # Calculate the LAR trapped-passing boundary
    LAR_tpb_O = np.array([])
    LAR_tpb_X = np.array([])

    if not bfield.plain_name == "LAR":
        print(
            "\nWARNING: Can not calculate LAR trapped-passing boundary analyticlly,"
            "for non LAR equilibrium. Returning empty LAR_tpb...\n"
        )
    else:
        # 0 <psi<psi_wall and at trapped - passing boundary E_O/μΒ0=B(theta=0,psi)/B0
        #                        and                        E_X/μΒ0=B(theta=pi,psi)/B0
        psis = np.linspace(0, psi_wallNU, config.Pzeta_density)
        LAR_tpb_O, _, _ = bfield.bigNU(psis, np.pi)
        LAR_tpb_X, _, _ = bfield.bigNU(psis, 0)

    return x, y_R, y_L, y_MA, LAR_tpb_O, LAR_tpb_X
