"""Function that draws figures depicting the B, i, g, E, Ψ quantities' contours in RZ
coordinates. It can also plot fixed points on certain plots."""

import matplotlib.pyplot as plt

from gcmotion.configuration.plot_parameters import RZBigContoursConfig
from gcmotion.plot._base._base_RZ_contour import _base_RZ_contour
from gcmotion.plot._base._base_fixed_points_profile_contour import _base_fixed_points_plot
from gcmotion.entities.profile import Profile

from gcmotion.utils.logger_setup import logger


def R_Z_contours(profile: Profile, **kwargs):
    r"""Plots the selected quantity's (:math:`\Psi`, E, B, I, g,
    :math:`\frac{\partial B}{\partial\theta}`, :math:`\frac{\partial B}{\partial\psi}`,
    :math:`\frac{\partial I}{\partial\psi}`, :math:`\frac{\partial g}{\partial\psi}`)
    contour plot in R, Z tokamak (cylindrical) coordinates, and in separate figures. Can
    also plot fixed points on certain contours.

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
        String of the form 'fp E b i g', 'fp E b i', 'fp E b g', 'fp E i g', 'fp E b', 'fp E i', 'fp E g'
        'fp b i g', 'fp b i', 'fp b g', 'fp i g', 'fp b', 'fp i', 'fp g', 'E b i g', E 'b i', 'E b g', 'E i g',
        'E b', 'E i', 'E g' 'b i g', 'b i', 'b g', 'i g', 'b', 'i', 'g'  (case insensitive)
        that determines which figures will be plotted, that of the Energy and magnetic flux,
        that of the magnetic field and its derivatives and/or that of the toroidal current and
        its derivative and/or that of the poloidal current and its derivatives. Defaults to 'E b i g'.

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

    # -----------------E, Ψ Figure--------------------

    if "e" in config.which.lower():
        # Create figure
        fig_kw = {
            "figsize": config.figsize_E_flux,
            "dpi": config.dpi,
            "layout": config.layout,
            "facecolor": config.facecolor,
        }

        fig_E = plt.figure(**fig_kw)

        tit_kw = {
            "fontsize": config.E_flux_suptitle_fontsize,
            "color": config.E_flux_suptitle_color,
        }

        fig_E.suptitle(f"Ψ & E in R-Z Coordinates ({plain_name})", **tit_kw)

        fig_e, fig_psi = fig_E.subfigures(1, 2)

        ax_e = fig_e.subplots(1, 1)
        ax_psi = fig_psi.subplots(1, 1)

        # -------------Ψ-------------

        _base_RZ_contour(
            profile=profile,
            fig=fig_psi,
            ax=ax_psi,
            which_Q="flux",
            units=config.flux_units,
            **kwargs,
        )

        ax_psi.set_title("Flux 'Ψ'")

        # -----------E---------------

        _base_RZ_contour(
            profile=profile,
            fig=fig_E,
            ax=ax_e,
            which_Q="E",
            units=config.E_units,
            locator="log",
            **kwargs,
        )

        ax_e.set_title("Energy")

        logger.info("Plotted E, Ψ in RZ_contours")

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

        # ------------B--------------

        _base_RZ_contour(
            profile=profile,
            fig=fig_b,
            ax=ax_b,
            which_Q="B",
            units=config.B_units,
            **kwargs,
        )

        ax_b.set_title("Magnetic Field 'B'")

        # -----------dB/dθ---------------

        _base_RZ_contour(
            profile=profile,
            fig=fig_dbdtheta,
            ax=ax_dbdtheta,
            which_Q="dbdtheta",
            units="",
            **kwargs,
        )

        ax_dbdtheta.set_title(r"$\partial B / \partial \theta$")

        # -----------dB/dψ---------------

        _base_RZ_contour(
            profile=profile,
            fig=fig_dbdpsi,
            ax=ax_dbdpsi,
            which_Q="dbdpsi",
            units="",
            **kwargs,
        )

        ax_dbdpsi.set_title(r"$\partial B / \partial \psi$")

        logger.info("Plotted B, dB_dtheta, dB_dpsi in RZ_contours")

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

        # -----------I---------------

        _base_RZ_contour(
            profile=profile,
            fig=fig_i,
            ax=ax_i,
            which_Q="I",
            units=config.I_units,
            **kwargs,
        )

        ax_i.set_title("Toroidal Current 'I'")

        # ----------dI/dψ----------------

        _base_RZ_contour(
            profile=profile,
            fig=fig_ider,
            ax=ax_ider,
            which_Q="ider",
            units="",
            **kwargs,
        )

        ax_ider.set_title(r"$\partial I / \partial \psi$")

        logger.info("Plotted I, dI_dpsi in RZ_contours")

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

        # ------------g--------------

        _base_RZ_contour(
            profile=profile,
            fig=fig_g,
            ax=ax_g,
            which_Q="g",
            units=config.g_units,
            **kwargs,
        )

        ax_g.set_title("Poloidal Current 'g'")

        # ----------dg/dψ----------------

        _base_RZ_contour(
            profile=profile,
            fig=fig_gder,
            ax=ax_gder,
            which_Q="gder",
            units="",
            **kwargs,
        )

        ax_gder.set_title(r"$\partial g / \partial \psi$")

        logger.info("Plotted g, dg_dpsi in RZ_contours")

    # -----------------Fixed Points Figure--------------------

    if "fp" in config.which.lower():
        # Create figure
        fig_kw = {
            "figsize": config.figsize_fp,
            "dpi": config.dpi,
            "layout": config.layout,
            "facecolor": config.facecolor,
        }

        fig_fp = plt.figure(**fig_kw)

        tit_kw = {
            "fontsize": config.fp_suptitle_fontsize,
            "color": config.fp_suptitle_color,
        }

        fig_fp.suptitle(f"Fixed Points Analysis in RZ Plain ({plain_name})", **tit_kw)

        fig_fp_E, fig_fp_dB = fig_fp.subfigures(1, 2)

        ax_fp_E = fig_fp_E.subplots(1, 1)
        ax_fp_dB = fig_fp_dB.subplots(1, 1)

        # ------------Fixed Points on Energy contour--------------

        _base_RZ_contour(
            profile=profile,
            fig=fig_fp_E,
            ax=ax_fp_E,
            which_Q="E",
            units=config.E_fp_units,
            locator="log",
            **kwargs,
        )

        _base_fixed_points_plot(
            profile=profile,
            ax=ax_fp_E,
            RZ_coords=config.fp_RZ_coords,
            **kwargs,
        )

        ax_fp_E.set_title("Fixed Points on Energy Contour", fontsize=11)

        # ----------Fixed Points on Stationary Curves ----------------

        _base_RZ_contour(
            profile=profile,
            fig=fig_fp_dB,
            ax=ax_fp_dB,
            which_Q="dbdtheta",
            units="",
            **kwargs,
        )

        _base_fixed_points_plot(
            profile=profile,
            ax=ax_fp_dB,
            RZ_coords=config.fp_RZ_coords,
            **kwargs,
        )

        ax_fp_dB.set_title("Fixed Points on Stationary Curves", fontsize=11)

        logger.info("Plotted Fixed Points in RZ_contours")

    plt.show()
