r"""
Calculates and plots the bifurcation diagram of the fixed points for multiple profiles
with different Pzetas or mus.
"""

import matplotlib.pyplot as plt
from time import time
from gcmotion.utils.logger_setup import logger
from gcmotion.entities.profile import Profile

from gcmotion.plot._base._bif_base._base_thetas_bif_plot import _thetas_bif_plot
from gcmotion.plot._base._bif_base._base_psis_bif_plot import _psis_bif_plot
from gcmotion.plot._base._bif_base._base_ndfp_bif_plot import _ndfp_bif_plot
from gcmotion.plot._base._bif_base._base_tpb_plot import _plot_trapped_passing_boundary
from gcmotion.scripts.fixed_points_bif.bifurcation import bifurcation

from gcmotion.entities.profile import Profile
from collections import deque
from gcmotion.configuration.plot_parameters import BifurcationPlotConfig


def bifurcation_plot(profile: Profile, COM_values: list | deque, **kwargs):
    r"""Draws the bifurcation diagrams for the :math:`theta`'s  fixed,
    the :math:`\psi`'s fixed and the number of fixed points found for
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
        plot_energy_bif : bool, optional
            Boolean determining weather the energy of each fixed point of each profile (each :math:`\P_{\zeta}`)
            is to be plotted. Defaults to ``False``.
        which_COM : str, optional
            Determines with regard to which COM (:math:`\mu` or :math:`P_{zeta}`) will the bifurcation
            analysis take place. Essentially determinies the independent variable on the axis' of the
            bifurcation diagram.
        energy_units : str, optional
            String specifying the unit of the calculated fixed points' energies. Defaults to ``"NUJoule"``.
        energies_info : bool, optional
            Boolean determining weather information on the fixed points' energies is to be is to be printed in the log.
            Defaults to ``True``.
        fp_only_confined : bool, optional
            Boolean determining if the search for :math:`\psi_{fixed}` will be conducted only for
            :math:`\psi` < :math:`\psi_{wall}` (confined particles). Defaults to ``False``.
    """

    # Unpack parameters
    config = BifurcationPlotConfig()
    for key, value in kwargs.items():
        setattr(config, key, value)

    start = time()
    # CAUTION: The bifurcation function takes in psis_fixed but returns P_thetas_fixed
    bifurcation_output = bifurcation(
        profile=profile,
        COM_values=COM_values,
        calc_energies=config.plot_energy_bif,
        **kwargs,
    )

    print(f"BIFURCATION RUN IN {(time() - start)/60:.1f} mins")

    logger.info(
        f"Ran bifurcation script for bifurcation plot with N={len(COM_values)} and for COM = {config.which_COM} in {(time() - start)/60:.1f} mins"
    )

    # Unpack bifurcation output
    X_thetas = bifurcation_output["X_thetas"]
    X_psis = bifurcation_output["X_psis"]
    O_thetas = bifurcation_output["O_thetas"]
    O_psis = bifurcation_output["O_psis"]
    num_of_XP = bifurcation_output["num_of_XP"]
    num_of_OP = bifurcation_output["num_of_OP"]
    X_energies = bifurcation_output["X_energies"]
    O_energies = bifurcation_output["O_energies"]

    # Create figure
    fig_kw = {
        "figsize": config.figsize,
        "dpi": config.dpi,
        "layout": config.layout,
        "facecolor": config.facecolor,
        "sharex": config.sharex,
    }

    selected_COMNU_str = config.which_COM + "NU"
    selected_COM_Q = getattr(profile, selected_COMNU_str, "PzetaNU")
    selected_COM_units = selected_COM_Q.units
    other_COM = _other_COM(profile=profile, COM=config.which_COM)

    num_of_subplots = 2

    if config.plot_ndfp:
        num_of_subplots = 3

    fig, ax = plt.subplots(num_of_subplots, 1, **fig_kw)
    plt.xlabel(
        _setup_x_label(config.which_COM) + f"[{selected_COM_units}]",
        fontsize=config.xlabel_fontsize,
    )
    fig.suptitle(
        "Fixed Points Bifurcation Diagram",
        fontsize=config.suptitle_fontsize,
        color=config.suptitle_color,
    )

    ax_theta = ax[0]
    ax_P_theta = ax[1]
    if config.plot_ndfp:
        ax_ndfp = ax[2]

    # Fixed thetas bifurcation diagram
    _thetas_bif_plot(
        profile=profile,
        COM_values=COM_values,
        X_thetas=X_thetas,
        O_thetas=O_thetas,
        ax=ax_theta,
        **kwargs,
    )

    logger.info(
        f"Made Xthetas, Othetas fixed bifurcation plot for COM = {config.which_COM} for {other_COM[0]} = {other_COM[1]}"
    )

    # P_theta Fixed Bifurcation
    _psis_bif_plot(
        profile=profile,
        COM_values=COM_values,
        X_psis=X_psis,
        O_psis=O_psis,
        ax=ax_P_theta,
        **kwargs,
    )

    logger.info(
        f"Made P_thetas fixed bifurcation plot for COM = {config.which_COM} for {other_COM[0]} = {other_COM[1]}"
    )

    if config.plot_ndfp:
        # Number of distinct fixed points Diagram
        _ndfp_bif_plot(
            profile=profile,
            COM_values=COM_values,
            num_of_XP=num_of_XP,
            num_of_OP=num_of_OP,
            ax=ax_ndfp,
            **kwargs,
        )

        logger.info(
            f"Made number of fixed points bifurcation plot for COM = {config.which_COM} for {other_COM[0]} = {other_COM[1]}"
        )

    if config.plot_energy_bif:
        _plot_trapped_passing_boundary(
            profile=profile,
            COM_values=COM_values,
            X_energies=X_energies,
            O_energies=O_energies,
            input_energy_units=config.energy_units,
            **kwargs,
        )

        logger.info(
            f"Made fixed points' energies bifurcation plot for COM = {config.which_COM} for {other_COM[0]} = {other_COM[1]}"
        )

    plt.ion()
    plt.show(block=True)


def _setup_x_label(which_COM: str):

    return r"${\mu}$" if which_COM == "mu" else r"$P_{\zeta}$"


def _other_COM(profile: Profile, COM: str):

    if COM == "mu":
        return "PzetaNU", getattr(profile, "PzetaNU")
    elif COM == "Pzeta":
        return "muNU", getattr(profile, "muNU")
