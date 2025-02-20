import contourpy
import numpy as np

from gcmotion.entities.profile import Profile
from gcmotion.configuration.scripts_configuration import ContourGeneratorConfig

tau = 2 * np.pi


def main_contour(profile: Profile, psilim: tuple, **kwargs):
    r"""[Step 1] Generate the main Profile Contour."""

    # Unpack parameters
    config = ContourGeneratorConfig()
    for key, value in kwargs.items():
        setattr(config, key, value)

    Q = profile.Q
    thetalim = Q((-tau, tau), "radians")
    psilim = Q(psilim, "psi_wall").to("NUMagnetic_flux")

    # Calculate Energy through psi and the convert psi to Ptheta
    psi_grid, theta_grid = np.meshgrid(
        np.linspace(psilim[0], psilim[1], config.main_grid_density),
        np.linspace(thetalim[0], thetalim[1], config.main_grid_density),
    )

    ptheta_grid = profile.findPtheta(
        psi=psi_grid,
        units="NUCanonical_momentum",
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
        y=ptheta_grid.m,
        z=energy_grid.m,
        line_type="Separate",
    )

    # We later need the Ptheta vertical span to validate objects.
    ylim = (ptheta_grid.m[0][0], ptheta_grid.m[0][-1])

    MainContour = {
        "C": C,
        "ylim": ylim,
        "energymin": energy_grid.min(),
        "energymax": energy_grid.max(),
    }

    return MainContour
