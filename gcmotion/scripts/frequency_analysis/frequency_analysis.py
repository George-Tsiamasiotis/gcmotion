import pandas as pd

from tqdm import tqdm
from dataclasses import asdict
from collections import deque

from numpy.typing import ArrayLike
from gcmotion.entities.profile import Profile

from gcmotion.configuration.scripts_configuration import (
    FrequencyAnalysisConfig,
)
from gcmotion.scripts.frequency_analysis.profile_analysis import (
    profile_analysis,
)


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

        # If an ArrayLike is passed, convert it to NU Quantity, else use
        # profile's value
        if muspan is not None:
            self.muspan = profile.Q(muspan, "NUMagnetic_moment")
        else:
            self.muspan = [profile.muNU]
        if Pzetaspan is not None:
            self.Pzetaspan = profile.Q(Pzetaspan, "NUCanonical_momentum")
        else:
            self.Pzetaspan = [profile.PzetaNU]
        if Espan is not None:
            self.Espan = profile.Q(Espan, "NUJoule")
        else:
            self.Espan = [profile.ENU]

    def start(self):
        # Progress bars
        # qkin and omegas calculations

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

        # This loop runs through all given parameters and returns all contour
        # orbits that managed to calculate their frequencies
        self.orbits = deque()
        for mu in self.muspan:
            pzeta_pbar.reset()
            for Pzeta in self.Pzetaspan:
                energy_pbar.reset()
                for E in self.Espan:

                    # Update profile
                    self.profile.muNU = mu
                    self.profile.PzetaNU = Pzeta
                    self.profile.ENU = E

                    # Profile Analysis returs either a list with found orbits,
                    # or None
                    result = profile_analysis(
                        profile=self.profile,
                        psilim=self.psilim,
                        **asdict(self.config),
                    )

                    if result is not None:
                        self.orbits += result

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
        }

        self.df = pd.DataFrame(d)
        return self.df

    def dump(self):
        pass
