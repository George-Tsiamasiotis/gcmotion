"""Function that draws figures depicting the B, i, g quantities' contours in RZ
coordinates"""

import re
import numpy as np
import matplotlib.pyplot as plt

from gcmotion.configuration.plot_parameters import RZBigContoursConfig
from gcmotion.plot.RZ_contour import R_Z_contour
from gcmotion.entities.profile import Profile

from gcmotion.utils.logger_setup import logger


def R_Z_big_contour(profile: Profile, **kwargs):
    r"""Plots the selected quantity's (B, I, g,
    :math:`\frac{\partial B}{\partial\theta}`, :math:`\frac{\partial B}{\partial\psi}`,
    :math:`\frac{\partial I}{\partial\psi}`, :math:`\frac{\partial g}{\partial\psi}`)
    contour plot in R, Z tokamak (cylindrical) coordinates, and in separate figures.

    Parameters
    ----------
    profile : Profile
        The Profile entity.
    Other Parameters
    ----------------
    parametric_density : int, optional
        Practiacally the density of the :math:`\theta`, :math:`\psi` contour
        meshgrid, from which the R, Z grid is calculated. Defults to 500.
    xmargin_perc : float, optional
        x-axis margin of xlim so that there is some blank (white) space in between the
        plot limits and each contour drawing. Defaults to 0.1.
    ymargin_perc : float, optional
        y-axis margin of ylim so that there is some blank (white) space in between the
        plot limits and each contour drawing. Defaults to 0.1.
    which : str
        String of the form 'b i g', 'b i', 'b g', 'i g', 'b', 'i', 'g' (case insensitive)
        that determines which figures will be plotted, that of the magnetic field and its
        derivatives and/or that of the toroidal current and its derivative and/or that of
        the poloidal current and its derivatives. Defaults to 'b i g'.

    Notes
    -----
    For a full list of all available optional parameters, see the dataclass
    RZBigContoursConfig at gcmotion/configuration/plot_parameters. The defaults values
    are set there, and are overwritten if passed as arguments.
    """

    logger.info("\t==> Plotting RZ big Contour...")

    # Unpack Parameters
    config = RZBigContoursConfig()
    for key, value in kwargs.items():
        setattr(config, key, value)

    plain_name = profile.bfield.plain_name

    # -----------------B Figure--------------------

    if "b" in config.which.lower():
        # Create figure
        fig_kw = {
            "figsize": config.figsize_B,
            "dpi": config.dpi,
            "layout": config.layout,
            "facecolor": config.facecolor,
        }

        fig_B = plt.figure(**fig_kw)

        tit_kw = {
            "fontsize": config.B_suptitle_fontsize,
            "color": config.B_suptitle_color,
        }

        fig_B.suptitle(f"B Profile in R-Z Coordinates ({plain_name})", **tit_kw)

        fig_b, fig_dbdtheta, fig_dbdpsi = fig_B.subfigures(1, 3)

        ax_b = fig_b.subplots(1, 1)
        ax_dbdtheta = fig_dbdtheta.subplots(1, 1)
        ax_dbdpsi = fig_dbdpsi.subplots(1, 1)

        R_Z_contour(
            profile=profile,
            fig=fig_b,
            ax=ax_b,
            which_Q="B",
            parametric_density=config.parametric_density,
            units=config.B_units,
            levels=config.levels,
            show=False,
        )

        ax_b.set_title("Magnetic Field 'B'")

        R_Z_contour(
            profile=profile,
            fig=fig_dbdtheta,
            ax=ax_dbdtheta,
            which_Q="dbdtheta",
            parametric_density=config.parametric_density,
            units="",
            levels=config.levels,
            show=False,
        )

        ax_dbdtheta.set_title(r"$\partial B / \partial \theta$")

        R_Z_contour(
            profile=profile,
            fig=fig_dbdpsi,
            ax=ax_dbdpsi,
            which_Q="dbdpsi",
            parametric_density=config.parametric_density,
            units="",
            levels=config.levels,
            show=False,
        )

        ax_dbdpsi.set_title(r"$\partial B / \partial \psi$")

        logger.info("Plotted B, dB_dtheta, dB_dpsi in RZ_big_contour")

    # -----------------I Figure--------------------

    if "i" in config.which.lower():
        # Create figure
        fig_kw = {
            "figsize": config.figsize_I,
            "dpi": config.dpi,
            "layout": config.layout,
            "facecolor": config.facecolor,
        }

        fig_I = plt.figure(**fig_kw)

        tit_kw = {
            "fontsize": config.I_suptitle_fontsize,
            "color": config.I_suptitle_color,
        }

        fig_I.suptitle(f"I Profile in R-Z Coordinates ({plain_name})", **tit_kw)

        fig_i, fig_ider = fig_I.subfigures(1, 2)

        ax_i = fig_i.subplots(1, 1)
        ax_ider = fig_ider.subplots(1, 1)

        R_Z_contour(
            profile=profile,
            fig=fig_i,
            ax=ax_i,
            which_Q="I",
            parametric_density=config.parametric_density,
            units=config.I_units,
            levels=config.levels,
            show=False,
        )

        ax_i.set_title("Toroidal Current 'I'")

        R_Z_contour(
            profile=profile,
            fig=fig_ider,
            ax=ax_ider,
            which_Q="ider",
            parametric_density=config.parametric_density,
            units="",
            levels=config.levels,
            show=False,
        )

        ax_ider.set_title(r"$\partial I / \partial \psi$")

        logger.info("Plotted I, dI_dpsi in RZ_big_contour")

    # -----------------g Figure--------------------
    if "g" in config.which.lower():
        # Create figure
        fig_kw = {
            "figsize": config.figsize_g,
            "dpi": config.dpi,
            "layout": config.layout,
            "facecolor": config.facecolor,
        }

        fig_g = plt.figure(**fig_kw)

        tit_kw = {
            "fontsize": config.g_suptitle_fontsize,
            "color": config.g_suptitle_color,
        }

        fig_g.suptitle(f"g Profile in R-Z Coordinates ({plain_name})", **tit_kw)

        fig_g, fig_gder = fig_g.subfigures(1, 2)

        ax_g = fig_g.subplots(1, 1)
        ax_gder = fig_gder.subplots(1, 1)

        R_Z_contour(
            profile=profile,
            fig=fig_g,
            ax=ax_g,
            which_Q="g",
            parametric_density=config.parametric_density,
            units=config.g_units,
            levels=config.levels,
            show=False,
        )

        ax_g.set_title("Poloidal Current 'I'")

        R_Z_contour(
            profile=profile,
            fig=fig_gder,
            ax=ax_gder,
            which_Q="gder",
            parametric_density=config.parametric_density,
            units="",
            levels=config.levels,
            show=False,
        )

        ax_gder.set_title(r"$\partial g / \partial \psi$")

        logger.info("Plotted g, dg_dpsi in RZ_big_contour")

    plt.show()
