r"""
Simple script that sets up the values of the desired quantities in a bifurcation plot.
"""

from gcmotion.entities.profile import Profile
from collections import deque
import numpy as np


def set_up_bif_plot_values(
    profile: Profile,
    COM_values: list | deque,
    y_values: list | deque | np.ndarray,
    which_COM: str,
    tilt_energies: bool = False,
    input_energy_units: str = "NUJoule",
) -> tuple[list, list]:
    r"""
    Simple function that sets up the values of the desired quantity in a bifurcation plot.
    Due to bifurcations an two or more y values may correspond to asingle x value.
    This function ensures that the plotted arrays have the same length.
    """

    set_up_x_values = []
    set_up_y_values = []

    # For the mu bifurcation we need to subtract mu*B0 from each energy
    if tilt_energies and which_COM == "mu":
        for i, COM_value in enumerate(COM_values):
            selected_com = profile.Q(COM_value, "NUmagnetic_moment")
            mu = selected_com
            muB0 = (mu * profile.Q(1, "NUTesla")).to(input_energy_units)
            y_list = y_values[i].m - muB0.m

            # Ensure y_list is iterable
            if np.isscalar(y_list):
                y_list = [y_list]

            set_up_x_values.extend([selected_com.m] * len(list(y_list)))
            set_up_y_values.extend(y_list)

    else:
        for i, COM_value in enumerate(COM_values):
            selected_com = COM_value
            y_list = y_values[i]

            # Ensure y_list is iterable
            if np.isscalar(y_list):
                y_list = [y_list]

            set_up_x_values.extend([selected_com] * len(list(y_list)))
            set_up_y_values.extend(y_list)

    return set_up_x_values, set_up_y_values
