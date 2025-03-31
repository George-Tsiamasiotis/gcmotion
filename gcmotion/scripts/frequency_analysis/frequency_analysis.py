r"""
=================
FrequencyAnalysis
=================

Class for calculating the frequencies of (μ, Ρζ, E) triplets inside a specific
Tokamak.

See documentation for a description of the algorithm.

"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from time import time
from copy import deepcopy
from termcolor import colored
from collections import deque

from matplotlib.patches import Patch
from gcmotion.entities.tokamak import Tokamak
from gcmotion.entities.profile import Profile

from . import triplet_analysis
from .triplet_analysis import (
    profile_triplet_analysis,
)
from .contour_orbit import ContourOrbit
from .contour_generators import main_contour
from gcmotion.utils.logger_setup import logger

from gcmotion.configuration.scripts_configuration import (
    FrequencyAnalysisPbarConfig,
    FrequencyAnalysisConfig,
)
from gcmotion.configuration.plot_parameters import (
    FrequencyAnalysisPlotConfig,
)

messages = {
    0: """
ERROR: Illegal input arrays' dimensions. One of the following restrictions must
be satisfied:
    - All 3 arrays are 1-dimensional (cartesian mode)
    - All 3 arrays are 2-dimensional with the same shape (matrix mode)
    - "muspan" and "Pzetaspan" are both 1-dimensional and "Espan" is None
      (dynamic minimum energy mode)
""",
    1: """
Input arrays seem to be orthogonal. Consider using only a single row/column
from each to activate cartesian mode, for significant perforance improvement.
""",
    2: """
WARNING: "skip_trapped" option is forced to True in dynamic energy minimum
mode.
""",
}


class FrequencyAnalysis:
    r"""Performs a Frequency Analysis on a given Tokamak, by calculating
    contours.

    Parameters
    ----------
    tokamak: Tokamak
        The tokamak to perform the analysis upon.
    psilim : tuple(float, float)
        The :math:`\psi` limit to restrict the search for contours, relative to
        :math:`\psi_{wall}`. Avoid setting the lower limit exactly 0, but set
        it at a very low value (e.g. 1e-7) to avoid RuntimeWarnings when ψ
        becomes negative due to rounding errors.
    muspan : numpy.ndarray
        The :math:`\mu` span. Can be either 1D or 2D. See documentation for
        definitions
    Pzetaspan : numpy.ndarray
        The :math:`P_\zeta` span. Can be either 1D or 2D. See documentation for
        definitions
    Espan : numpy.ndarray
        The Energy span. Can be either 1D or 2D. See documentation for
        definitions

    Other Parameters
    ----------------
    main_grid_density: int, optional
        The Main Contour's' grid density. For analytical equilibria, a minimum
        of 500 is recommended, while numerical equilibria require at least
        1000. This option mostly affects trapped orbits around O-points.
        Diminishing results are observed after 1800. Defaults to 1000.
    local_grid_density: int, optional
        The local contours' grid density. 100 seems to work best for almost all
        cases. Defaults to 100.
    theta_expansion, psi_expansion: float, optional
        The bounding box expansion factor around which to create the local
        contour. Note that the expanded bounding box is still limited by the
        Main Contour's limits. Default to 1.2.
    pzeta_min_step: float, optional
        The minimum :math:`P_\zeta` step allowed. This step is usually smaller
        than the relative minimum step, but takes over for :math:`P_\zeta
        \approx 0`. Defaults to 2e-4.
    passing_pzeta_rstep: float, optional
        The relative :math:`P_\zeta` step size for passing orbits. Defaults to
        1e-3.
    trapped_pzeta_rstep: float, optional
        The relative :math:`P_\zeta` step size for trapped orbits. Defaults to
        1e-3.
    energy_rstep: float, optional
        The relative E step size. Defaults to 1e-3.
    skip_trapped: bool, optional
        Whether or not to skip calculations of trapped orbits. Defaults to
        False.
    skip_passing: bool, optional
        Whether or not to skip calculations of passing orbits. Defaults to
        False.
    skip_copassing: bool, optional
        Whether or not to skip calculations of co-passing orbits. Defaults to
        False.
    skip_cupassing: bool, optional
        Whether or not to skip calculations of cu-passing orbits. Defaults to
        False.
    calculate_omega_theta: bool, optional
        Whether or not to calculate each orbit's :math:`\omega_\theta`.
        Defaults to True.
    calculate_qkinetic: bool, optional
        Whether or not to calculate each orbit's :math:`q_{kinetic}`. Defaults
        to True.
    cocu_classification: bool, optional
        Whether or not to further classify passing orbits as Co-/Cu-passing.
        Defaults to True.
    min_vertices_method_switch: int, optional
        Minimum number of vertices below which the frequency calculations
        switch to double contour mode. Defaults to 40.
    max_pzeta_method_switch: float, optional
        Maximum :math:`P_\zeta` (absolute) value, below which the frequency
        calculations switch to double contour mode. Defaults to 0.
    relative_upper_E_factor: float, optional
        Used only in dynamic minimum energy mode. Defines the upper Espan limit
        to search for trapped orbits, relative to the total minimum Energy.
        Defaults to 1.1.
    logspace_len: int, optional
        Used only in dynamic minimum energy mode. The number of Energies for
        which to search trapped orbits, up until ``relative_upper_E_factor``.
        Defaults to 50.
    trapped_min_num: int, optional
        Used only in dynamic minimum energy mode. Defines the minimum number of
        trapped orbits, after which calculation stops. Defaults to 1.

    Examples
    --------

    >>> import numpy as np
    >>> import gcmotion as gcm
    >>>
    >>> Rnum = 1.65
    >>> anum = 0.5
    >>> B0num = 1
    >>> species = "p"
    >>> Q = gcm.QuantityConstructor(R=Rnum, a=anum, B0=B0num, species=species)
    >>>
    >>> B0 = Q(B0num, "Tesla")
    >>> i = Q(0, "NUPlasma_current")
    >>> g = Q(1, "NUPlasma_current")
    >>>
    >>> tokamak = gcm.Tokamak(
    ...     R=Q(Rnum, "meters"),
    ...     a=Q(anum, "meters"),
    ...     qfactor=gcm.qfactor.Unity(),
    ...     bfield=gcm.bfield.LAR(B0=B0, i=i, g=g),
    ...     efield=gcm.efield.Nofield(),
    ... )
    >>>
    >>> # Cartesian mode span
    >>> muspan = np.array([1e-5])
    >>> Pzetaspan = np.linspace(-0.04, -0.005, 10)
    >>> Espan = np.logspace(7.8e-6, 4e-5, 20)
    >>>
    >>> freq = gcm.FrequencyAnalysis(
    ...     tokamak=tokamak,
    ...     psilim=(1e-7, 1.2),
    ...     muspan=muspan,
    ...     Pzetaspan=Pzetaspan,
    ...     Espan=Espan,
    ...     main_contour_density=800,
    ... )
    >>>
    >>> freq.start()
    >>> print(freq)  # doctest: +SKIP
    >>> freq.scatter(x="Pzeta", y="qkinetic")  # doctest: +SKIP
    >>> df = freq.to_dataframe()
    """

    def __init__(
        self,
        tokamak: Tokamak,
        psilim: tuple,
        muspan: np.ndarray,
        Pzetaspan: np.ndarray,
        Espan: np.ndarray = None,
        method: str = "contour",
        **kwargs,
    ):
        logger.info("==> Setting up Frequency Analysis...")

        # Unpack kwargs
        self.config = FrequencyAnalysisConfig()
        for key, value in kwargs.items():
            setattr(self.config, key, value)

        self.profile = Profile(
            tokamak=tokamak,
            species=tokamak.Q.args["species"],
        )
        self.psilim = self.profile.Q(psilim, "psi_wall").to("NUmf").m

        self._process_arguements(muspan, Pzetaspan, Espan, method)
        self.analysis_completed = False
        log_init(self.config)

    def _process_arguements(
        self,
        muspan: np.ndarray,
        Pzetaspan: np.ndarray,
        Espan: np.ndarray,
        method: str,
    ):
        r"""
        Decides the iteration mode depending on the shape of the input arrays.

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
        self.method = method

        # Select Mode
        match (self.muspan, self.Pzetaspan, self.Espan):
            case np.ndarray(), np.ndarray(), np.ndarray() if (
                self.muspan.ndim == self.Pzetaspan.ndim == self.Espan.ndim == 1
            ):
                self.mode = "cartesian"
                self.triplets_num = (
                    self.muspan.shape[0]
                    * self.Pzetaspan.shape[0]
                    * self.Espan.shape[0]
                )
            case np.ndarray(), np.ndarray(), np.ndarray() if (
                self.muspan.shape == self.Pzetaspan.shape == self.Espan.shape
            ):
                self.mode = "matrix"
                self.triplets_num = self.muspan.size  # Same for all 3
                # if (
                #     np.linalg.det(self.muspan)
                #     == np.linalg.det(self.Espan)
                #     == np.linalg.det(self.Pzetaspan)
                #     == 0.0
                # ):
                #     logger.warning(messages[1])
                #     warnings.warn(messages[1])
            case np.ndarray(), np.ndarray(), None if (
                self.muspan.ndim == self.Pzetaspan.ndim == 1
            ):
                self.mode = "dynamicEmin"
                self.triplets_num = (
                    self.muspan.shape[0] * self.Pzetaspan.shape[0]
                )
            case _:
                logger.error(messages[0])
                raise ValueError(messages[0])

        if self.method.lower() in ("orbit", "orbits"):
            self.method = "orbits"
        if self.method.lower() in ("contour", "contours"):
            self.method = "contour"

        log_method_selection(self.mode, self.config)

    def start(self, pbar: bool = True):
        r"""Iterates through the triplets and finds the frequencies of the
        resulting found contours.

        Parameters
        ----------
        pbar: bool, optional
            Whether or not to display a progress bar. Defaults to True.

        """
        logger.info("==> Beginning Frequency Analysis.")
        start = time()
        self.orbits = []

        match self.mode:
            case "cartesian":
                self._start_cartesian(pbar=pbar)
            case "matrix":
                self._start_matrix(pbar=pbar)
            case "dynamicEmin":
                self._start_dynamicEmin(pbar=pbar)

        self.trapped_orbits_num = len(
            [orb for orb in self.orbits if orb.trapped]
        )
        self.passing_orbits_num = len(
            [orb for orb in self.orbits if orb.passing]
        )
        self.copassing_orbits_num = len(
            [orb for orb in self.orbits if orb.copassing]
        )
        self.cupassing_orbits_num = len(
            [orb for orb in self.orbits if orb.cupassing]
        )
        self.undefined_orbits_num = len(
            [orb for orb in self.orbits if orb.undefined]
        )
        logger.info(f"\tFound {self.trapped_orbits_num} trapped orbits.")
        logger.info(f"\tFound {self.passing_orbits_num} passing orbits.")
        logger.info(f"\tFound {self.copassing_orbits_num} copassing orbits.")
        logger.info(f"\tFound {self.cupassing_orbits_num} cupassing orbits.")

        duration = self.profile.Q(time() - start, "seconds")
        logger.info(f"--> Frequency Analysis Complete. Took {duration:.4g~#P}")

        self.analysis_completed = True

    def _triplet_analysis(
        self, main_contour: dict, profile: Profile
    ) -> list[ContourOrbit]:

        if self.method == "contour":
            return profile_triplet_analysis(
                main_contour=main_contour,
                profile=profile,
                psilim=self.psilim,
                config=self.config,
            )

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

                MainContour = main_contour(profile, self.psilim, self.config)

                for E in self.Espan:
                    profile.ENU = profile.Q(E, "NUJoule")

                    # =========================================================
                    # Profile Analysis returs either a list with found orbits,
                    # or an empty list.
                    found_orbits = self._triplet_analysis(
                        main_contour=MainContour,
                        profile=profile,
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

        self.single_contour_orbits, self.double_contour_orbits = (
            log_contour_method()
        )

    def _start_matrix(self, pbar: bool):
        r"""Matrix Method: Used if all input arrays are 2D and of the same
        shape."""

        assert self.muspan.shape == self.Pzetaspan.shape == self.Espan.shape

        rows, columns = self.muspan.shape  # All spans have the same shape
        grid = np.array(
            (self.muspan, self.Pzetaspan.T, self.Espan.T)
        ).T.reshape(rows * columns, 3)

        # Progress bar
        pbars = _ProgressBars(pbar=pbar)
        matrix_pbar = pbars.matrix_pbar(total=grid.shape[0])

        profile = deepcopy(self.profile)
        self.orbits = deque()

        # Even though its slower, we have to generate the main contour again
        # for every (mu, Pzeta, E) triplet, since we don't know if the Pzeta-E
        # grid is orthogonal (it usually is not). If they are, it's better to
        # use cartesian mode.
        for mu, Pzeta, E in grid:
            profile.muNU = profile.Q(mu, "NUMagnetic_moment")
            profile.PzetaNU = profile.Q(Pzeta, "NUCanonical_momentum")
            profile.ENU = profile.Q(E, "NUJoule")

            MainContour = main_contour(profile, self.psilim, self.config)

            found_orbits = self._triplet_analysis(
                main_contour=MainContour,
                profile=profile,
            )
            # Avoid floating point precision errors
            for orb in found_orbits:
                orb.mu = mu
                orb.Pzeta = Pzeta
                orb.E = E
            self.orbits += found_orbits

            matrix_pbar.update()

        self.single_contour_orbits, self.double_contour_orbits = (
            log_contour_method()
        )

    def _start_dynamicEmin(self, pbar: bool):
        r"""Dynamic minimum energy Method: Used if muspan and Pzetaspan are 1D
        and Espan is None."""

        if self.config.skip_trapped:
            self.config.skip_trapped = False
            logger.warning(messages[2])
            warnings.warn(messages[2])

        pbars = _ProgressBars(pbar=pbar)
        mu_pbar = pbars.mu_pbar(total=len(self.muspan))
        pzeta_pbar = pbars.pzeta_pbar(total=len(self.Pzetaspan))

        profile = deepcopy(self.profile)
        self.orbits = deque()

        # This loop runs through all given parameters and returns all contour
        # orbits that managed to calculate their frequencies
        for mu in self.muspan:
            pzeta_pbar.reset()
            profile.muNU = profile.Q(mu, "NUMagnetic_moment")

            for Pzeta in self.Pzetaspan:
                profile.PzetaNU = profile.Q(Pzeta, "NUCanonical_momentum")

                MainContour = main_contour(
                    profile,
                    self.psilim,
                    calculate_min=True,
                    config=self.config,
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
                    found_orbits = self._triplet_analysis(
                        main_contour=MainContour,
                        profile=profile,
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

        self.single_contour_orbits, self.double_contour_orbits = (
            log_contour_method()
        )

    def _start_dynamicEmin_opoints(self, pbar: bool):
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
                "color": pd.Series([orb.color for orb in self.orbits]),
            }

        self.df = pd.DataFrame(d)
        return self.df

    def scatter(self, x: str, y: str, scatter_kw: dict = {}):
        r"""Draws a scatter plot of the calculated frequencies.

        Parameters
        ----------

        x: {"mu", "Pzeta", "E", "qkinetic", "omega_theta", "omega_zeta"}
            The x coordinate, as a column name of the Dataframe.
        y: {"mu", "Pzeta", "E", "qkinetic", "omega_theta", "omega_zeta"}
            The y coordinate, as a column name of the Dataframe.
        scatter_kw: dict, optional
            Further arguements to be passed to matplotlib's scatter method.
            Defaults to {}.
        """

        if not hasattr(self, "df"):
            self.to_dataframe()

        config = FrequencyAnalysisPlotConfig()

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
        } | scatter_kw
        ax.scatter(xs, ys, c=colors, **scatter_kw)
        ax.set_xlabel(_scatter_labels(x), size=12)
        ax.set_ylabel(_scatter_labels(y), size=12)
        ax.tick_params(axis="both", size=15)
        ax.set_title(ax.get_xlabel() + " - " + ax.get_ylabel(), size=15)
        ax.legend(handles=[trapped, copassing, cupassing, undefined])
        ax.grid(True)

        # Add a horizontal line to y=0
        if config.add_hline:
            ax.axhline(y=0, ls="--", lw=1.5, c="k")

        plt.show()

    def hexbin():
        # TODO:
        pass

    def alpha_tricontour():
        # TODO:
        pass

    def __str__(self):
        # Parameters availiable before run().
        string = ""
        if self.config.print_tokamak:
            string += "\n" + self.profile.tokamak.__str__()

        string += colored("\nFrequency Analysis\n", "green")
        string += (
            f"{"Mode":>28} : {colored(self.mode.capitalize(), "light_blue")}\n"
            f"{"Particle species":>28} : "
            f"{colored(self.profile.species_name, "light_blue"):<16}\n"
            f"{"Triplets number":>28} : {self.triplets_num}\n"
        )

        if not self.analysis_completed:
            return string

        string += (
            f"{"Total Orbits Found":>28} : {len(self.orbits):<16}\n"
            f"{"Trapped/Passing Orbits Found":>28} : "
            + "".join(
                (
                    f"trapped: {self.trapped_orbits_num}",
                    " / ",
                    f"passing: {self.passing_orbits_num:<16}\n",
                )
            )
            + f"{"Co/Cu-Passing Orbits Found":>28} : "
            + "".join(
                (
                    f"CoPassing: {self.copassing_orbits_num}",
                    " / ",
                    f"CuPassing: {self.cupassing_orbits_num:<16}\n",
                )
            )
            + f"{"Undefined Orbits Found":>28} : "
            + "".join((f"Undefined: {self.undefined_orbits_num:<16}\n",))
            + f"{"Single/Double Contour Orbits":>28} : "
            + "".join(
                (
                    f"Single: {self.single_contour_orbits}",
                    " / ",
                    f"Double: {self.double_contour_orbits:<16}\n",
                )
            )
        )
        return string


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
            desc=self.config.tqdm_matrix_desc,
            unit=self.config.tqdm_matrix_unit,
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


def log_init(config: FrequencyAnalysisConfig):
    r"""Logs Frequency Analysis options after passed by the user."""
    logger.debug(f"\tpzeta_min_step = {config.pzeta_min_step}")
    logger.debug(f"\tpassing_pzeta_rtol = {config.passing_pzeta_rstep}")
    logger.debug(f"\ttrapped_pzeta_rtol = {config.passing_pzeta_rstep}")
    logger.debug(f"\tenergy_rstep= {config.energy_rstep}")
    logger.debug(f"\tqkinetic_cutoff = {config.qkinetic_cutoff}")
    logger.debug(f"\tskip_trapped = {config.skip_trapped}")
    logger.debug(f"\tskip_passing = {config.skip_passing}")
    logger.debug(f"\tcocu_classification = {config.cocu_classification}")
    logger.debug(f"\tcalculate_qkin = {config.calculate_qkinetic}")
    logger.debug(f"\tcalculate_omega_theta = {config.calculate_omega_theta}")
    logger.debug(
        f"\tmethod_switch_threshold = {config.max_vertices_method_switch}"
    )


def log_method_selection(mode: str, config: FrequencyAnalysisConfig):
    logger.info(f"\tMode: {mode}")
    match mode:
        case "cartesian":
            logger.debug(f"\tmain_grid_density = {config.main_grid_density}")
            logger.debug(f"\tlocal_grid_density = {config.local_grid_density}")
        case "dynamicEmin":
            logger.debug(f"\tlogspace_len = {config.logspace_len}")
            logger.debug(
                "\trelative_upper_E_factor = "
                f"{config.relative_upper_E_factor}"
            )
        case _:
            pass


def log_contour_method():
    r"""Logs the module-level variables that keep track of for how many orbits
    each method was called.
    """
    logger.info(
        "\tFrequencies calculated with single contouring: "
        f"{triplet_analysis.single_contour_orbits}"
    )
    logger.info(
        "\tFrequencies calculated with double contouring: "
        f"{triplet_analysis.double_contour_orbits}"
    )

    return_tuple = (
        triplet_analysis.single_contour_orbits,
        triplet_analysis.double_contour_orbits,
    )

    # Reset them to allow many frequency analysis to run without restarting the
    # interpreter
    triplet_analysis.single_contour_orbits = 0
    triplet_analysis.double_contour_orbits = 0

    return return_tuple
