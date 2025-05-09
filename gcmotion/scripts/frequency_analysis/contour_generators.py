r"""
==================
Contour Generators
==================

Functions that generate the Main and local Contours.

"""

import contourpy
import numpy as np

from gcmotion.entities.profile import Profile
from gcmotion.scripts.frequency_analysis.contour_orbit import ContourOrbit
from gcmotion.configuration.scripts_configuration import (
    FrequencyAnalysisConfig,
)

tau = 2 * np.pi


def main_contour(
    profile: Profile,
    psilim: tuple,
    config: FrequencyAnalysisConfig,
    calculate_min: bool = False,
):
    r"""Generate the main Profile Contour.

    Psi limits are the full span, as called from FrequencyAnalysis.

    Returns
    -------
    dict
        A dict containing the contour generator and the psilim of the energy
        grid.
    """

    psi_grid, theta_grid = np.meshgrid(
        np.linspace(psilim[0], psilim[1], config.main_grid_density),
        np.linspace(-tau, tau, config.main_grid_density),
    )

    energy_grid = profile._findEnergyNU(
        psi=psi_grid,
        theta=theta_grid,
        potential=True,
    )

    # NOTE: Use line_type="Separate". Then C.lines(level) returns a list of
    # (N,2) numpy arrays containing the vertices of each contour line.
    C = contourpy.contour_generator(
        x=theta_grid,
        y=psi_grid,
        z=energy_grid,
        line_type="Separate",
    )

    MainContour = {
        "C": C,
        "psilim": psilim,
    }
    if calculate_min:
        MainContour |= {"zmin": energy_grid.min()}

    return MainContour


def local_contour(
    profile: Profile, orbit: ContourOrbit, config: FrequencyAnalysisConfig
):
    r"""Expands bounding box and creates a local contour generator inside of
    it.

    Returns
    -------
    dict
        A dict containing the contour generator and the psilim of the energy
        grid.

    """

    # Expand theta but dont let it out of (-tau, tau)
    (xmin, ymin), (xmax, ymax) = orbit.bbox
    if orbit.trapped:
        thetamean = (xmin + xmax) / 2
        thetaspan = (xmax - xmin) / 2
        thetamin = max(thetamean - config.theta_expansion * thetaspan, -np.pi)
        thetamax = min(thetamean + config.theta_expansion * thetaspan, np.pi)
    elif orbit.passing:
        thetamin, thetamax = -tau, tau

    # Expand psilim without limiting it (but staying positive), to catch all
    # orbits. psilim is already in NU so don't bother with Quantities..
    psimean = (ymin + ymax) / 2
    psispan = (ymax - ymin) / 2
    psimin = max(psimean - config.psi_expansion * psispan, 1e-8)
    psimax = psimean + config.psi_expansion * psispan

    psi_grid, theta_grid = np.meshgrid(
        np.linspace(psimin, psimax, config.local_grid_density),
        np.linspace(thetamin, thetamax, config.local_grid_density),
    )

    energy_grid = profile._findEnergyNU(
        psi=psi_grid,
        theta=theta_grid,
        potential=True,
    )

    # NOTE: Use line_type="Separate". Then C.lines(level) returns a list of
    # (N,2) numpy arrays containing the vertices of each contour line.
    C = contourpy.contour_generator(
        x=theta_grid,
        y=psi_grid,
        z=energy_grid,
        line_type="Separate",
    )

    LocalContour = {
        "C": C,
        "psilim": (psimin, psimax),
    }

    return LocalContour
