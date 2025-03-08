import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from copy import deepcopy
from collections import Counter, deque
from dataclasses import asdict
from pint.registry import Quantity

from numpy.typing import ArrayLike
from matplotlib.patches import Patch
from gcmotion.entities.profile import Profile

from gcmotion.configuration.scripts_configuration import (
    FrequencyAnalysisConfig,
)
from gcmotion.configuration.plot_parameters import (
    FrequencyAnalysisPlotConfig,
)
from gcmotion.scripts.frequency_analysis.profile_analysis import (
    profile_analysis,
)
from gcmotion.scripts.frequency_analysis.contour_generators import main_contour


class FrequencyAnalysis:

    def __init__(
        self,
        profile: Profile,
        psilim: tuple,
        muspan: ArrayLike = None,
        Pzetaspan: ArrayLike = None,
        Espan: ArrayLike = None,
        **kwargs,
    ):

        # Unpack kwargs
        self.config = FrequencyAnalysisConfig()
        for key, value in kwargs.items():
            setattr(self.config, key, value)

        self.profile = profile
        self.psilim = (
            profile.Q(psilim, "psi_wall").to("NUMagnetic_flux").magnitude
        )

        self._process_arguements(muspan, Pzetaspan, Espan)

    def _process_arguements(
        self,
        muspan: ArrayLike,
        Pzetaspan: ArrayLike,
        Espan: ArrayLike,
    ):
        r"""
        Matrix mode
        -----------
        If all 3 spans are 2d arrays with the same shape, create a grid and
        iterate through every (i,j,k).
            If one and only one is not passed, create a 2d tile grid with the
            shape of the other 2 with the profile's property.

        Cartesian Mode
        --------------
        If all 3 spans are 1d arrays, iterate through their cartesian product.
            If 2 (at most) are not passed, use the profile's property.
        """
        self.muspan = muspan
        self.Pzetaspan = Pzetaspan
        self.Espan = Espan

        # Make sure all spans are numpy arrays
        for span in ("muspan", "Pzetaspan", "Espan"):
            if isinstance(getattr(self, span), np.ndarray):
                pass
            if getattr(self, span) is None:
                profile_property = getattr(
                    self.profile, span[:-4] + "NU"
                ).magnitude
                setattr(self, span, np.array([profile_property]))

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
            case _:
                raise ValueError("Illegal Input")

        # Quantify
        self.muspan = self.profile.Q(self.muspan, "NUMagnetic_moment")
        self.Pzetaspan = self.profile.Q(self.Pzetaspan, "NUCanonical_momentum")
        self.Espan = self.profile.Q(self.Espan, "NUJoule")

    def start(self, pbar: bool = True):
        if self.mode == "cartesian":
            self._start_cartesian(pbar=pbar)

    def _start_cartesian(self, pbar: bool):
        # Progress bars
        # qkin and omegas calculations
        self.config.tqdm_enable = pbar

        global_pbar_kw = {  # Shared through all 3 colour bars
            "ascii": self.config.tqdm_ascii,
            "colour": self.config.tqdm_colour,
            "dynamic_ncols": self.config.tqdm_dynamic_ncols,
            "disable": not self.config.tqdm_enable,
        }

        mu_pbar = tqdm(
            position=0,
            total=len(self.muspan),
            desc=self.config.tqdm_mu_desc,
            unit=self.config.tqdm_mu_unit,
            **global_pbar_kw,
        )
        pzeta_pbar = tqdm(
            position=1,
            total=len(self.Pzetaspan),
            desc=self.config.tqdm_pzeta_desc,
            unit=self.config.tqdm_pzeta_unit,
            **global_pbar_kw,
        )
        energy_pbar = tqdm(
            position=2,
            total=len(self.Espan),
            desc=self.config.tqdm_energy_desc,
            unit=self.config.tqdm_energy_unit,
            **global_pbar_kw,
        )

        profile = deepcopy(self.profile)
        # This loop runs through all given parameters and returns all contour
        # orbits that managed to calculate their frequencies
        self.orbits = deque()
        for mu in self.muspan:
            pzeta_pbar.reset()
            profile.muNU = mu

            for Pzeta in self.Pzetaspan:
                energy_pbar.reset()
                profile.PzetaNU = Pzeta
                MainContour = main_contour(profile, self.psilim)

                for E in self.Espan:
                    profile.ENU = E

                    # Profile Analysis returs either a list with found orbits,
                    # or None
                    self.orbits += profile_analysis(
                        main_contour=MainContour,
                        profile=profile,
                        psilim=self.psilim,
                        **asdict(self.config),
                    )

                    energy_pbar.update()
                pzeta_pbar.update()
            mu_pbar.update()

        # Refresh them
        for pbar in (mu_pbar, pzeta_pbar, energy_pbar):
            pbar.refresh()

    def to_dataframe(self):

        d = {
            "Energy": pd.Series([orb.E for orb in self.orbits]),
            "Pzeta": pd.Series([orb.Pzeta for orb in self.orbits]),
            "mu": pd.Series([orb.mu for orb in self.orbits]),
            "qkinetic": pd.Series([orb.qkinetic for orb in self.orbits]),
            "omega_theta": pd.Series([orb.omega_theta for orb in self.orbits]),
            "omega_zeta": pd.Series([orb.omega_zeta for orb in self.orbits]),
            "orbit_type": pd.Series([orb.string for orb in self.orbits]),
            # "color": pd.Series([orb.color for orb in self.orbits]),
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

        scatter_kw = {
            "s": config.scatter_size,
        }
        ax.scatter(xs, ys, c=colors, **scatter_kw)
        ax.axhline(y=0, ls="--", lw=1.5, c="k")
        ax.set_xlabel(_scatter_labels(x))
        ax.set_ylabel(_scatter_labels(y))
        ax.set_title(ax.get_xlabel() + " - " + ax.get_ylabel())
        ax.legend(handles=[trapped, copassing, cupassing, undefined])
        ax.grid(True)
        plt.show()

    def dump(self):
        pass


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


def _create_init_array(value: ArrayLike | Quantity):

    try:
        value.shape
    except AttributeError:  # default to profile's property
        array = np.array(value.magnitude)
        print("foo1")
    else:
        array = value
        print("foo2")

    assert array.shape == ()
    return array
