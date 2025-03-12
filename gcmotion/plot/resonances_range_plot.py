r"""Simplpe script that draws the resonance range diagram, (omega max at O point) for
each :math:`\mu` or :math:`P_{\zeta}`."""

import pandas as pd
import matplotlib.pyplot as plt
from time import time

from collections import deque
from gcmotion.utils.logger_setup import logger
from gcmotion.entities.profile import Profile

from gcmotion.scripts.resοnance_range import omegas_max
from gcmotion.scripts.fixed_points_bif.bif_values_setup import set_up_bif_plot_values

from gcmotion.configuration.plot_parameters import ResRangePlotConfig


def res_range_plot(profile: Profile, COM_values: list | deque, **kwargs):
    r"""Draws the resonance range diagram, (omega max at O point) for
    each :math:`\mu` or :math:`P_{\zeta}`.

    :meta public:

        Parameters
        ----------
        profile : Profile
            Profile object containing Tokamak information.
        COM_values : list, deque
            List of COM values :math:`P_{\zeta}`'s or :math:`\mu`'s in [NU].
        Other Parameters
        ----------
        thetalim : list, optional
            Limits of the of the :math:`\theta`, :math:`\psi` search area with respect
            to the :math:`\theta` variable. Defaults to [-:math:`\pi`, :math:`\pi`].
        psilim : list, optional
            Limits of the of the :math:`\theta`, :math:`\psi` search area with respect
            to the :math:`\psi` variable. Defaults to [0.01 , 1.8]. CUTION: The limits are given
            normalized to :math:`\psi_{wall}`.
        method : str, optional
            String that indicates which method will be used to find the systems fixed
            points in :py:func:`single_fixed_point`. Can either be "fsolve" (deterministic)
            or "differential evolution" (stochastic). Defaults to "fsolve".
        dist_tol : float, optional
            Tolerance below which two fixed points are not considered distinct. The differences between
            both :math:`\theta` and :math:`\psi` of the fixed points must be below this tolerance for
            the fixed points to be considered the same. Defaults to 1e-3.
        fp_ic_scan_tol : float, optional
            Tolerance below which the sum of the squares of the time derivatives of the
            :math:`\theta` and :math:`\psi` variavles is considered zero. It is passed into
            :py:func:`fp_ic_scan`. Defaults to 5 * 1e-8.
        ic_theta_grid_density : int, optional
            Density of the :math:`\theta`, :math:`\psi` 2D grid to be scanned for initial conditiond
            (fixed points candidates) with respect to the :math:`\theta` variable. It is passed into
            :py:func:`fp_ic_scan` Defaults to 400.
        ic_psi_grid_density : int, optional
            Density of the :math:`\theta`, :math:`\psi` 2D grid to be scanned for initial conditiond
            (fixed points candidates) with respect to the :math:`\psi` variable. It is passed into
            :py:func:`fp_ic_scan` Defaults to 400.
        random_fp_init_cond : bool, optional
            Boolean determining weather random initial conditions are to be used instead of those
            provided by :py:func:`fp_ic_scan`. Defaults to ``False``.
        fp_info : bool, optional
            Boolean determining weather fixed points' information is to be is to be printed in the log. Defaults to ``False``.
        bif_info: bool, optional
            Boolean that determines weather information regarding the bifurcation process is to
            be is to be printed in the log. Defaults to ``False``.
        fp_ic_info : bool, optional
            Boolean determing weather information on the initial condition is to be is to be printed in the log.
            Defaults to ``False``.
        which_COM : str, optional
            Determines with regard to which COM (:math:`\mu` or :math:`P_{zeta}`) will the resonance range
            analysis take place. Essentially determinies the independent variable on the axis' of the
            bifurcation diagram.
        fp_only_confined : bool, optional
            Boolean determining if the search for :math:`\psi_{fixed}` will be conducted only for
            :math:`\psi` < :math:`\psi_{wall}` (confined particles). Defaults to ``False``.
        freq_units : str
            String that determines what will be the frequency units of the y-axis.
            Defaults to 'NUw0'.

    """
    logger.info("\t==> Plotting Base Number of distinct Fixed Points Bifurcation Diagram...")

    # Unpack parameters
    config = ResRangePlotConfig()
    for key, value in kwargs.items():
        setattr(config, key, value)

    start = time()

    omegas = omegas_max(profile=profile, COM_values=COM_values, **kwargs)

    print(f"OMEGAS MAX RUN IN {(time() - start)/60:.1f} mins")

    # Here which_COM is always Pzeta because it has to do with how the values are handled
    # inside set_up_bif_plot_values
    COM_plot, omegas_plot = set_up_bif_plot_values(
        profile=profile, COM_values=COM_values, y_values=omegas, which_COM="Pzeta"
    )

    # Create figure
    fig_kw = {
        "figsize": config.figsize,
        "dpi": config.dpi,
        "layout": config.layout,
        "facecolor": config.facecolor,
    }

    fig, ax = plt.subplots(1, 1, **fig_kw)

    selected_COMNU_str = config.which_COM + "NU"
    selected_COM_Q = getattr(profile, selected_COMNU_str, "PzetaNU")
    selected_COM_units = selected_COM_Q.units

    ax.set_title(
        rf"Island Resonance Range - $\omega_\theta$ at O Points ({profile.bfield.plain_name})",
        fontsize=config.titlesize,
        color=config.titlecolor,
    )

    ax.scatter(
        COM_plot,
        omegas_plot,
        marker=config.marker_style,
        color=config.marker_color,
        s=config.marker_size,
    )

    xlabel_COM = _set_xlabel(which_COM=config.which_COM)
    ax.set_xlabel(
        f"{xlabel_COM} [{selected_COM_units}]",
        fontsize=config.xlabel_fontsize,
        rotation=config.xlabel_rotation,
    )

    ax.set_ylabel(
        r"$\omega_\theta^{max}$" + f" [{config.freq_units}]",
        fontsize=config.ylabel_fontsize,
        rotation=config.ylabel_rotation,
    )

    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    # df = pd.read_pickle(r"C:\Users\georg\Downloads\max_omega_theta_per_Pzeta_df")
    # # print(df)
    # df = df.reset_index()
    # ax.scatter(df.iloc[:, 0], abs(df.iloc[:, 1]), marker="x", color="b")

    plt.ion()
    plt.show(block=True)


def _set_xlabel(which_COM: str) -> str:

    return r"$P_{\zeta}$" if which_COM == "Pzeta" else r"$\mu$"
