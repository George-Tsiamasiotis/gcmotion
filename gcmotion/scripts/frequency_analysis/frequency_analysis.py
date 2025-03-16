r"""
==================
Frequency Analysis
==================

The FrequencyAnalysis class iterates through (μ, Pζ, Ε) values upon a given
Profile, and finds the ωθ, ωζ frequencies and their ratio qkinetic by searching
for contours.

Each contour represents a specific family of orbits represented by the same 3
Constants of Motion, and differing only in their initial conditions. By
exploiting the fact that our poloidal angle is in fact Boozer theta, the area
contained within the contour is equal to 2π*Jθ, where Jθ the corresponding
action variable. We then use the definitions:

ωθ = dE/dJθ

qkin = -dJθ/dJζ = -dJθ/dPζ
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from time import time
from copy import deepcopy
from collections import deque
from dataclasses import asdict

from numpy.typing import ArrayLike
from matplotlib.patches import Patch
from gcmotion.entities.profile import Profile

from .profile_analysis import profile_analysis
from .contour_generators import main_contour
from gcmotion.utils.logger_setup import logger

from gcmotion.configuration.scripts_configuration import (
    FrequencyAnalysisPbarConfig,
    FrequencyAnalysisConfig,
)
from gcmotion.configuration.plot_parameters import (
    FrequencyAnalysisPlotConfig,
)


class FrequencyAnalysis:
    r"""Performs a Frequency Analysis on a given Profile, by calculating
    closed contours.

    Parameters
    ----------
    profile : Profile
        The Profile to perform the analysis upon.
    psilim : tuple
        The :math:`\psi` limit to restrict the search for contours, relative to
        :math:`\psi_{wall}`.
    muspan : np.ndarray
        The :math:`\mu` span. See Notes section for definitions.
    Pzetaspan : np.ndarray
        The :math:`P_\zeta` span. See Notes section for definitions.
    Espan : np.ndarray
        The Energy span. See Notes section for definitions.

    Notes
    -----
    The algorithm supports 3 modes:

    1. Cartesian Mode: Activated by passing 3 1D arrays.
        The algorithm takes all combinations (cartesian product) of every array
        entry. If you want to iterate through only 1 COM value, use
        np.array([<value>]).

    2. Matrix Mode: Activated by passing 3 2D arrays, **with the same shape**
        The algorithm creates triplets of COMs by stacking the 3 arrays. Each
        triplet is defined as (muspan[i,j], Pzetaspan[i,j], Espan[i,j]), where
        0<=i<nrows and 0<=j<ncols. Useful when the grid is not orthogonal, for
        example when analysing a certain :math:`P_\zeta - E` domain of the
        parabolas.

    3. Dynamic minimum energy Mode: Activated by only passing muspan and
        Pzetspan as 1D arrays. The algorithm finds the minimum vaule of the
        energy grid for every (mu, Pzeta) pair, which is always found at an
        O-point, and slowly increments it until it finds 1 trapped orbit. This
        orbit's frequency is (very close to) the O-point frequency, which is
        always the highest frequency of this specific family of trapped orbits.
        This frequency defines the maximum frequency with which the particles
        resonate, with 0 being the slowest (separatrix). This mode can only
        find the O-point frequency on the total minumum energy. If more
        O-points are present, then we must use method 4.

    4. Dynamic O-point minimum energy Mode: To be implemented

    """

    def __init__(
        self,
        profile: Profile,
        psilim: tuple,
        muspan: ArrayLike,
        Pzetaspan: ArrayLike,
        Espan: ArrayLike = None,
        **kwargs,
    ):
        logger.info("==> Setting Frequency Analysis")

        # Unpack kwargs
        self.config = FrequencyAnalysisConfig()
        for key, value in kwargs.items():
            setattr(self.config, key, value)

        self.psilim = profile.Q(psilim, "psi_wall").to("NUMagnetic_flux").m
        logger.debug(f"\tpsilim = {self.psilim}")

        self.profile = profile
        # COM values must be explicitly defined
        if not (profile.mu is profile.Pzeta is profile.E is None):
            msg = "Warning: Profile initial COMs are ignored."
            logger.warning(msg)
            warnings.warn(msg)

        self._process_arguements(muspan, Pzetaspan, Espan)

    def _process_arguements(
        self,
        muspan: ArrayLike,
        Pzetaspan: ArrayLike,
        Espan: ArrayLike,
    ):
        r"""
        Cartesian Mode
        --------------
        If all 3 spans are 1d arrays, iterate through their cartesian product.
            If 2 (at most) are not passed, use the profile's property.

        Matrix mode
        -----------
        If all 3 spans are 2d arrays with the same shape, create a grid and
        iterate through every (i,j,k).
            If one and only one is not passed, create a 2d tile grid with the
            shape of the other 2 with the profile's property.

        Dynamic minimum energy Mode
        ---------------------------
        If muspan and Pzetaspan are 1d, arrays, iterate trough their cartesian
        product. For each pair, find the minimum energy grid from the energy
        grid, and slowly increment it until we find 1 trapped orbit, which ends
        the loop.

        """
        self.muspan = muspan
        self.Pzetaspan = Pzetaspan
        self.Espan = Espan

        # Select Mode
        match (self.muspan, self.Pzetaspan, self.Espan):
            case np.ndarray(), np.ndarray(), np.ndarray() if (
                self.muspan.ndim == self.Pzetaspan.ndim == self.Espan.ndim == 1
            ):
                self.mode = "cartesian"
            case np.ndarray(), np.ndarray(), np.ndarray() if (
                self.muspan.shape == self.Pzetaspan.shape == self.Espan.shape
            ):
                self.mode = "matrix"
            case np.ndarray(), np.ndarray(), None if (
                self.muspan.ndim == self.Pzetaspan.ndim == 1
            ):
                self.mode = "dynamicEmin"
            case _:
                raise ValueError("Illegal Input")

        logger.info(f"\tMode: {self.mode}")

    def start(self, pbar: bool = True):
        r"""Calculates the frequencies

        Parameters
        ----------
        pbar: bool, optional
            Whether or not to display a progress bar. Defaults to True.

        """
        logger.info("==> Beginning Frequency Analysis.")
        start = time()

        match self.mode:
            case "cartesian":
                self._start_cartesian(pbar=pbar)
            case "matrix":
                self._start_matrix(pbar=pbar)
            case "dynamicEmin":
                self._start_dynamicEmin(pbar=pbar)

        duration = self.profile.Q(time() - start, "seconds")
        logger.info(f"--> Frequency Analysis Complete. Took {duration:.4g~#P}")

    def _start_cartesian(self, pbar: bool):
        r"""Cartesian Method: Used if all input arrays are 1D."""

        profile = deepcopy(self.profile)
        self.orbits = deque()

        pbars = _ProgressBars(pbar=pbar)
        mu_pbar = pbars.mu_pbar(total=len(self.muspan))
        pzeta_pbar = pbars.pzeta_pbar(total=len(self.Pzetaspan))
        energy_pbar = pbars.energy_pbar(total=len(self.Espan))

        # This loop runs through all given parameters and returns all contour
        # orbits that managed to calculate their frequencies
        for mu in self.muspan:
            pzeta_pbar.reset()
            profile.muNU = profile.Q(mu, "NUMagnetic_moment")

            for Pzeta in self.Pzetaspan:
                energy_pbar.reset()
                profile.PzetaNU = profile.Q(Pzeta, "NUCanonical_momentum")

                MainContour = main_contour(profile, self.psilim)
                # Emin = MainContour["zmin"]
                # self.Espan = np.logspace(
                #     np.log10(Emin), np.log10(Emin * 1.1), 50
                # )

                for E in self.Espan:
                    profile.ENU = profile.Q(E, "NUJoule")

                    # =========================================================
                    # Profile Analysis returs either a list with found orbits,
                    # or None
                    found_orbits = profile_analysis(
                        main_contour=MainContour,
                        profile=profile,
                        psilim=self.psilim,
                        **asdict(self.config),
                    )

                    # Avoid floating point precision errors
                    for orb in found_orbits:
                        orb.mu = mu
                        orb.Pzeta = Pzeta
                        orb.E = E
                    self.orbits += found_orbits
                    # =========================================================

                    energy_pbar.update()
                pzeta_pbar.update()
            mu_pbar.update()

        # Refresh them
        for pbar in (mu_pbar, pzeta_pbar, energy_pbar):
            pbar.refresh()

    def _start_matrix(self, pbar: bool):
        r"""Cartesian Method: Used if all input arrays are 2D and of the same
        shape."""

        assert self.muspan.shape == self.Pzetaspan.shape == self.Espan.shape

        rows, columns = self.muspan.shape  # All spans have the same shape
        grid = np.array(
            (self.muspan, self.Pzetaspan.T, self.Espan.T)
        ).T.reshape(rows * columns, 3)

        # Progress bar
        pbars = _ProgressBars(pbar=pbar)
        matrix_pbar = pbars.mu_pbar(total=grid.shape[0])

        profile = deepcopy(self.profile)
        self.orbits = deque()

        # Even though its slower, we have to generate the main contour again
        # for every (mu, Pzeta, E) triplet, since we don't know if the Pzeta-E
        # grid is orthogonal (it usually is not).
        for mu, Pzeta, E in grid:
            profile.muNU = profile.Q(mu, "NUMagnetic_moment")
            profile.PzetaNU = profile.Q(Pzeta, "NUCanonical_momentum")
            profile.ENU = profile.Q(E, "NUJoule")

            MainContour = main_contour(profile, self.psilim)

            found_orbits = profile_analysis(
                main_contour=MainContour,
                profile=profile,
                psilim=self.psilim,
                **asdict(self.config),
            )
            # Avoid floating point precision errors
            for orb in found_orbits:
                orb.mu = mu
                orb.Pzeta = Pzeta
                orb.E = E
            self.orbits += found_orbits

            matrix_pbar.update()

        matrix_pbar.refresh()

    def _start_dynamicEmin(self, pbar: bool):
        r"""Cartesian Method: Used if all input arrays are 1D."""

        if self.config.skip_trapped:
            msg = (
                "'skip_trapped' option must be False when using "
                + "'dynamic Emin' method. "
            )
            logger.error(msg)
            raise RuntimeError(msg)

        profile = deepcopy(self.profile)
        self.orbits = deque()

        pbars = _ProgressBars(pbar=pbar)
        mu_pbar = pbars.mu_pbar(total=len(self.muspan))
        pzeta_pbar = pbars.pzeta_pbar(total=len(self.Pzetaspan))

        # This loop runs through all given parameters and returns all contour
        # orbits that managed to calculate their frequencies
        for mu in self.muspan:
            pzeta_pbar.reset()
            profile.muNU = profile.Q(mu, "NUMagnetic_moment")

            for Pzeta in self.Pzetaspan:
                profile.PzetaNU = profile.Q(Pzeta, "NUCanonical_momentum")

                MainContour = main_contour(
                    profile, self.psilim, calculate_min=True
                )
                Emin = MainContour["zmin"]
                self.Espan = np.logspace(
                    np.log10(Emin),
                    np.log10(Emin * self.config.relative_upper_E_factor),
                    self.config.logspace_len,
                )

                for E in self.Espan:
                    # =========================================================
                    profile.ENU = profile.Q(E, "NUJoule")

                    # Profile Analysis returs either a list with found orbits,
                    # or None
                    found_orbits = profile_analysis(
                        main_contour=MainContour,
                        profile=profile,
                        psilim=self.psilim,
                        **asdict(self.config),
                    )

                    # Avoid floating point precision errors
                    for orb in found_orbits:
                        orb.mu = mu
                        orb.Pzeta = Pzeta
                        orb.E = E
                    self.orbits += found_orbits

                    if len(found_orbits) >= self.config.trapped_min_num:
                        break
                    # =========================================================

                pzeta_pbar.update()
            mu_pbar.update()

        # Refresh them
        for pbar in (mu_pbar, pzeta_pbar):
            pbar.refresh()

    def _start_dynamicEmin_opoints(self, pbar: bool):
        # TODO:
        pass

    def results(self):
        # TODO:
        pass

    def to_dataframe(self, extended: bool = False):
        r"""Creates a pandas DataFrame with the resulting frequencies.

        Parameters
        ----------
        extended : bool
            Whether or not to add extra information for every orbit.
        """
        d = {
            "Energy": pd.Series([orb.E for orb in self.orbits]),
            "Pzeta": pd.Series([orb.Pzeta for orb in self.orbits]),
            "mu": pd.Series([orb.mu for orb in self.orbits]),
            "qkinetic": pd.Series([orb.qkinetic for orb in self.orbits]),
            "omega_theta": pd.Series([orb.omega_theta for orb in self.orbits]),
            "omega_zeta": pd.Series([orb.omega_zeta for orb in self.orbits]),
            "orbit_type": pd.Series([orb.string for orb in self.orbits]),
        }
        if extended:
            d |= {
                "area": pd.Series([orb.area for orb in self.orbits]),
                "Jtheta": pd.Series([orb.Jtheta for orb in self.orbits]),
                "Jzeta": pd.Series([orb.Jzeta for orb in self.orbits]),
                "edge": pd.Series([orb.edge_orbit for orb in self.orbits]),
                "color": pd.Series([orb.color for orb in self.orbits]),
            }

        self.df = pd.DataFrame(d)
        return self.df

    def scatter(self, x: str, y: str, **kwargs):

        config = FrequencyAnalysisPlotConfig()
        for key, value in kwargs.items():
            setattr(config, key, value)

        # Manual lengend entries patches
        trapped = Patch(color=config.trapped_color, label="Trapped")
        copassing = Patch(color=config.copassing_color, label="Co-passing")
        cupassing = Patch(color=config.cupassing_color, label="Cu-Passing")
        undefined = Patch(color=config.undefined_color, label="Undefined")

        fig_kw = {
            "figsize": config.scatter_figsize,
            "dpi": config.scatter_dpi,
            "layout": "constrained",
        }
        fig = plt.figure(**fig_kw)
        ax = fig.add_subplot()

        xs, ys = self.df[x], self.df[y]
        colors = tuple(orb.color for orb in self.orbits)

        # Overwrite defaults and pass all kwargs to scatter
        scatter_kw = {
            "s": config.scatter_size,
        }
        ax.scatter(xs, ys, c=colors, **scatter_kw)
        ax.set_xlabel(_scatter_labels(x), size=12)
        ax.set_ylabel(_scatter_labels(y), size=12)
        ax.tick_params(axis="both", size=15)
        ax.legend(handles=[trapped, copassing, cupassing, undefined])
        ax.grid(True)

        # Add a second legend with the value of the COM that doesn't change, if
        # it exists:
        annotation = ""
        for com in ("mu", "Pzeta", "Energy"):
            value = self.df[com].unique()
            if len(value) == 1:
                annotation += f", Fixed {com} = {value}"
        ax.set_title(
            ax.get_xlabel() + " - " + ax.get_ylabel() + annotation, size=15
        )

        # Add a horizontal line to y=0
        if config.add_hline:
            ax.axhline(y=0, ls="--", lw=1.5, c="k")

        plt.show()

    def dump(self):
        # TODO:
        pass


# =============================================================================


class _ProgressBars:
    r"""Creates a progress bar for each COM."""

    def __init__(self, pbar: bool = True):
        self.config = FrequencyAnalysisPbarConfig()

        self.global_pbar_kw = {  # Shared through all 3 colour bars
            "ascii": self.config.tqdm_ascii,
            "colour": self.config.tqdm_colour,
            "smoothing": self.config.tqdm_smoothing,
            "dynamic_ncols": self.config.tqdm_dynamic_ncols,
            "disable": not pbar or not self.config.tqdm_enable,
        }

    def mu_pbar(self, total: int, position=0):
        return tqdm(
            position=position,
            total=total,
            desc=self.config.tqdm_mu_desc,
            unit=self.config.tqdm_mu_unit,
            **self.global_pbar_kw,
        )

    def pzeta_pbar(self, total: int, position=1):
        return tqdm(
            position=position,
            total=total,
            desc=self.config.tqdm_pzeta_desc,
            unit=self.config.tqdm_pzeta_unit,
            **self.global_pbar_kw,
        )

    def energy_pbar(self, total: int, position=2):
        return tqdm(
            position=position,
            total=total,
            desc=self.config.tqdm_energy_desc,
            unit=self.config.tqdm_energy_unit,
            **self.global_pbar_kw,
        )

    def matrix_pbar(self, total: int, position=0):
        return tqdm(
            position=position,
            total=total,
            desc=self.config.tqdm_energy_desc,
            unit=self.config.tqdm_energy_unit,
            **self.global_pbar_kw,
        )


def _scatter_labels(index: str):
    titles = {
        "Energy": r"$Energy [NU]$",
        "Pzeta": r"$P_\zeta [NU]$",
        "mu": r"$\mu [NU]$",
        "qkinetic": r"$q_{kinetic}$",
        "omega_theta": r"$\omega_\theta [\omega_0]$",
        "omega_zeta": r"$\omega_\zeta [\omega_0]$",
    }
    return titles[index]
