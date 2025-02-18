import pint
import numpy as np

from time import time
from copy import deepcopy
from contourpy import contour_generator

from gcmotion.utils.logger_setup import logger
from gcmotion.entities.profile import Profile
from gcmotion.configuration.scripts_configuration import ContourFreqConfig

tau = 2 * np.pi
Q = pint.get_application_registry().Quantity


def _create_contours(
    profile: Profile, psilim: tuple, config: ContourFreqConfig
) -> dict:

    contours: dict = _create_main_contour(profile, psilim, config)
    contours |= _create_dpzeta_contours(profile, psilim, config)

    return contours


def _create_main_contour(
    profile: Profile, psilim: tuple, config: ContourFreqConfig
) -> dict:
    r"""Creates a ContourGenerator from contourpy, as well as a couple of
    needed quantities."""

    logger.info("\tCreating Contour Generator...")
    start = time()

    thetalim = profile.Q((-tau, tau), "radians")
    psilim = profile.Q(psilim, "psi_wall").to("NUMagnetic_flux")
    psi_grid, theta_grid = np.meshgrid(
        np.linspace(psilim[0], psilim[1], config.grid_density),
        np.linspace(thetalim[0], thetalim[1], config.grid_density),
    )

    energy_grid = profile.findEnergy(
        psi=psi_grid,
        theta=theta_grid.m,
        units="NUJoule",
        potential=config.potential,
    )
    ptheta_grid = profile.findPtheta(
        psi=psi_grid,
        units="NUCanonical_momentum",
    )

    C = contour_generator(
        x=theta_grid.m,
        y=ptheta_grid.m,
        z=energy_grid.m,
        line_type="Separate",
        fill_type="OuterCode",
    )
    energy_span = (energy_grid.min().m, energy_grid.max().m)
    ylim = (ptheta_grid.m[0][0], ptheta_grid.m[0][-1])

    # Extra quantities we will need, but C has no __dict__
    metadata = {"energy_span": energy_span, "ylim": ylim}

    dur = Q(time() - start, "seconds")
    logger.info(f"\tTook {dur:.4g~#P}.")

    Contour = {
        "EnergyContour": C,
        "EnergyMetadata": metadata,
    }
    return Contour


def _create_dpzeta_contours(
    profile: Profile, psilim: tuple, config: ContourFreqConfig
) -> dict:

    # Copy them, otherwise they point to the same instance
    lower_profile = deepcopy(profile)
    upper_profile = deepcopy(profile)

    # Update Pzetas
    lower_profile.PzetaNU = lower_profile.PzetaNU * (1 - config.pzeta_rtol)
    upper_profile.PzetaNU = upper_profile.PzetaNU * (1 + config.pzeta_rtol)

    logger.info("\tCreating Pzeta Contour Generators...")
    start = time()

    contours_list = []
    metadata_list = []
    for prof in (lower_profile, upper_profile):

        thetalim = prof.Q((-tau, tau), "radians")
        psilim = prof.Q(psilim, "psi_wall").to("NUMagnetic_flux")
        psi_grid, theta_grid = np.meshgrid(
            np.linspace(psilim[0], psilim[1], config.grid_density),
            np.linspace(thetalim[0], thetalim[1], config.grid_density),
        )

        energy_grid = prof.findEnergy(
            psi=psi_grid,
            theta=theta_grid.m,
            units="NUJoule",
            potential=config.potential,
        )
        ptheta_grid = prof.findPtheta(
            psi=psi_grid,
            units="NUCanonical_momentum",
        )

        C = contour_generator(
            x=theta_grid.m,
            y=ptheta_grid.m,
            z=energy_grid.m,
            line_type="Separate",
            fill_type="OuterCode",
        )
        energy_span = (energy_grid.min().m, energy_grid.max().m)
        ylim = (ptheta_grid.m[0][0], ptheta_grid.m[0][-1])

        # Extra quantities we will need, but C has no __dict__
        metadata = {"energy_span": energy_span, "ylim": ylim}

        contours_list.append(C)
        metadata_list.append(metadata)

    Contours = {
        "PzetaLowerProfile": lower_profile,
        "PzetaUpperProfile": upper_profile,
        "PzetaLower": lower_profile.PzetaNU.m,
        "PzetaUpper": upper_profile.PzetaNU.m,
        "PzetaLowerContour": contours_list[0],
        "PzetaUpperContour": contours_list[1],
        "PzetaLowerMetadata": metadata_list[0],
        "PzetaUpperMetadata": metadata_list[1],
    }

    dur = Q(time() - start, "seconds")
    logger.info(f"\tTook {dur:.4g~#P}.")

    return Contours
