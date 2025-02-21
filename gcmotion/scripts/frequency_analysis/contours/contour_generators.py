import contourpy
import numpy as np
import matplotlib.pyplot as plt

from gcmotion.entities.profile import Profile
from gcmotion.configuration.scripts_configuration import ContourGeneratorConfig
from gcmotion.scripts.frequency_analysis.contours.contour_orbits import (
    ContourOrbit,
)

tau = 2 * np.pi


def main_contour(profile: Profile, psilim: tuple, **kwargs):
    r"""[Step 1] Generate the main Profile Contour."""

    # Unpack parameters
    config = ContourGeneratorConfig()
    for key, value in kwargs.items():
        setattr(config, key, value)

    thetalim = profile.Q((-tau, tau), "radians")
    psilim = profile.Q(psilim, "psi_wall").to("NUMagnetic_flux")

    psi_grid, theta_grid = np.meshgrid(
        np.linspace(psilim[0], psilim[1], config.main_grid_density),
        np.linspace(thetalim[0], thetalim[1], config.main_grid_density),
    )

    energy_grid = profile.findEnergy(
        psi=psi_grid,
        theta=theta_grid.m,
        units="NUjoule",
    )

    # NOTE: Use line_type="Separate". Then C.lines(level) returns a list of
    # (N,2) numpy arrays containing the vertices of each contour line.
    C = contourpy.contour_generator(
        x=theta_grid.m,
        y=psi_grid.m,
        z=energy_grid.m,
        line_type="Separate",
    )

    MainContour = {
        "C": C,
        "psilim": psilim.to("NUMagnetic_flux").m,
        # "energymin": energy_grid.min(),
        # "energymax": energy_grid.max(),
    }

    return MainContour


def local_contour(profile: Profile, orbit: ContourOrbit):
    r"""Expands bounding box and creates a local contour generator inside of
    it."""

    config = ContourGeneratorConfig()

    # Expand theta and psi grids but dont let theta out of (-tau, tau) or psi
    # out of psiwall_lim
    # TODO: use xmin/ymax to make it more readable
    bbox = orbit.bbox
    if orbit.trapped:
        thetamean = (bbox[0][0] + bbox[1][0]) / 2
        thetaspan = (bbox[1][0] - bbox[0][0]) / 2
        thetamin = max(thetamean - config.theta_expansion * thetaspan, -tau)
        thetamax = min(thetamean + config.theta_expansion * thetaspan, tau)
    elif orbit.passing:
        thetamin, thetamax = -tau, tau

    psimean = (bbox[0][1] + bbox[1][1]) / 2
    psispan = bbox[1][1] - bbox[0][1]
    psilim = profile.Q(profile.psiwall_lim, "psi_wall").to("NUmagnetic_flux")
    psimin = max(psimean - config.psi_expansion * psispan / 2, psilim[0].m)
    psimax = min(psimean + config.psi_expansion * psispan / 2, psilim[1].m)

    # Calculate local Contour
    thetalim = profile.Q((thetamin, thetamax), "radians")
    psilim = profile.Q((psimin, psimax), "NUmagnetic_flux")

    psi_grid, theta_grid = np.meshgrid(
        np.linspace(psilim[0], psilim[1], config.local_grid_density),
        np.linspace(thetalim[0], thetalim[1], config.local_grid_density),
    )

    energy_grid = profile.findEnergy(
        psi=psi_grid,
        theta=theta_grid.m,
        units="NUjoule",
    )

    # NOTE: Use line_type="Separate". Then C.lines(level) returns a list of
    # (N,2) numpy arrays containing the vertices of each contour line.
    C = contourpy.contour_generator(
        x=theta_grid.m,
        y=psi_grid.m,
        z=energy_grid.m,
        line_type="Separate",
    )

    LocalContour = {
        "C": C,
        "psilim": psilim.to("psi_wall").m,
        # "energymin": energy_grid.min(),
        # "energymax": energy_grid.max(),
    }

    return LocalContour
