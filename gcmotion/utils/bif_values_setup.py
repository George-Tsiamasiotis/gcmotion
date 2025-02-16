r"""
Simple script that sets up the values of the desired quantities in a bifurcation plot.
"""

from collections import deque
import numpy as np


def set_up_bif_plot_values(
    profiles: list | deque,
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
    selected_COMNU_str = which_COM + "NU"

    # For the mu bifurcation we need to subtract mu*B0 from each energy
    if tilt_energies and which_COM == "mu":
        for i, profile in enumerate(profiles):
            selected_com = getattr(profile, selected_COMNU_str, "PzetaNU")
            mu = profile.muNU
            muB0 = (mu * profile.Q(1, "NUTesla")).to(input_energy_units)
            y_list = y_values[i].m - muB0.m

            # Ensure y_list is iterable
            if np.isscalar(y_list):
                y_list = [y_list]

            set_up_x_values.extend([selected_com] * len(list(y_list)))
            set_up_y_values.extend(y_list)

    else:
        for i, profile in enumerate(profiles):
            selected_com = getattr(profile, selected_COMNU_str, "PzetaNU")
            y_list = y_values[i]

            # Ensure y_list is iterable
            if np.isscalar(y_list):
                y_list = [y_list]

            set_up_x_values.extend([selected_com] * len(list(y_list)))
            set_up_y_values.extend(y_list)

    return set_up_x_values, set_up_y_values
