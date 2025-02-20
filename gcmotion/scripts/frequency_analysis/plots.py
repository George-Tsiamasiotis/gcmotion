import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Patch
from matplotlib.collections import LineCollection

from gcmotion.entities.profile import Profile
from gcmotion.plot.profile_contour import profile_energy_contour
from gcmotion.scripts.frequency_analysis.contours.contour_orbits import (
    ContourOrbit,
)


def _plot_results(paths: ContourOrbit, config):

    # Setup figure
    fig_kw = {
        "figsize": config.figsize,
        "dpi": config.dpi,
        "layout": config.layout,
    }
    fig = plt.figure(**fig_kw)

    ax_dict = fig.subplot_mosaic(
        [
            ["paths", "qkin"],
            ["thetas", "zetas"],
        ],
        per_subplot_kw={
            "paths": {"title": "Contour Orbits"},
            "qkin": {"title": r"$q_{kinetic} - E[NU]$"},
            "thetas": {"title": r"$\omega_\theta [\omega_0] - E[NU]$"},
            "zetas": {"title": r"$\omega_\zeta[\omega_0] - E[NU]$"},
        },
    )

    # Manual lengend entries patches
    trapped = Patch(color=config.trapped_color, label="Trapped")
    copassing = Patch(color=config.copassing_color, label="Co-passing")
    cupassing = Patch(color=config.cupassing_color, label="Counter-Passing")
    undefined = Patch(color=config.undefined_color, label="Undefined")

    #################
    # Contour Paths #
    #################

    # See LineCollection example
    # Unpacking and creating a LineCollection is MUCH faster.
    path_vertices = tuple((np.column_stack(path.vertices.T) for path in paths))
    colors = tuple(line.color for line in paths)

    collection = LineCollection(path_vertices, colors=colors)
    ax_dict["paths"].add_collection(collection)
    ax_dict["paths"].set_xlim([-np.pi, np.pi])
    ax_dict["paths"].set_ylim(paths[1].ylim)
    ax_dict["paths"].set_title(r"Contour Segments")
    ax_dict["paths"].set_xlabel(r"$\theta [radians]$")
    ax_dict["paths"].set_ylabel(r"$P_\theta [NU]$")

    ax_dict["paths"].legend(
        handles=[trapped, copassing, cupassing, undefined], loc="upper right"
    )

    #############
    # q-kinetic #
    #############

    qs = tuple(path.qkinetic for path in paths)
    energies = tuple(path.E for path in paths)
    colors = tuple(path.color for path in paths)
    ax_dict["qkin"].scatter(energies, qs, c=colors, s=config.scatter_size)

    ax_dict["qkin"].axhline(y=0, ls="--", lw=0.5, c="k")
    ax_dict["qkin"].set_xlabel("Energy [NU]")
    ax_dict["qkin"].set_ylabel(r"$q_{kinetic}$")
    ax_dict["qkin"].grid(True, zorder=-2)

    ax_dict["qkin"].legend(handles=[trapped, copassing, cupassing, undefined])
    ax_dict["qkin"].tick_params(axis="x", labelrotation=40, labelsize=9)

    ################
    # omega_thetas #
    ################

    omegas = tuple(path.omega_theta for path in paths)
    energies = tuple(path.E for path in paths)
    colors = tuple(path.color for path in paths)
    ax_dict["thetas"].scatter(
        energies, omegas, c=colors, s=config.scatter_size
    )

    ax_dict["thetas"].axhline(y=0, ls="--", lw=0.5, c="k")
    ax_dict["thetas"].set_xlabel("Energy [NU]")
    ax_dict["thetas"].set_ylabel(r"$\omega_\theta [\omega_0]$")
    ax_dict["thetas"].tick_params(axis="x", labelrotation=30, labelsize=9)
    ax_dict["thetas"].grid(True, zorder=-2)

    ax_dict["thetas"].legend(
        handles=[trapped, copassing, cupassing, undefined]
    )

    ###############
    # omega_zetas #
    ###############

    omegas = tuple(path.omega_zeta for path in paths)
    energies = tuple(path.E for path in paths)
    colors = tuple(path.color for path in paths)
    ax_dict["zetas"].scatter(energies, omegas, c=colors, s=config.scatter_size)

    ax_dict["zetas"].axhline(y=0, ls="--", lw=0.5, c="k")
    ax_dict["zetas"].set_xlabel("Energy [NU]")
    ax_dict["zetas"].set_ylabel(r"$\omega_\zeta [\omega_0]$")
    ax_dict["zetas"].tick_params(axis="x", labelrotation=40, labelsize=9)
    ax_dict["zetas"].grid(True, zorder=-2)

    ax_dict["zetas"].legend(handles=[trapped, copassing, cupassing, undefined])

    plt.show()


# =============================== Debug Plots ===============================


def debug_plot_valid_orbits(profile: Profile, orbits: list):
    r"""Prints Profile Contour with all valid isoenergy orbits found"""

    ax = profile_energy_contour(
        profile,
        psilim=profile.psilim,
        E_units="NUJoule",
        flux_units="NUMagnetic_flux",
        canmom_units="NUCanonical_momentum",
        show=False,
    )

    for orbit in orbits:
        ax.plot(*orbit.vertices.T, color="red", zorder=10)

    plt.show()
    plt.close()
