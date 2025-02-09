import numpy as np

from gcmotion.configuration.plot_parameters import AutoYspan
from gcmotion.utils.logger_setup import logger


def _pi_mod(x, lim: list = [-np.pi, np.pi]) -> tuple[np.ndarray, list]:
    """Mods an array between [0,2π] or [-π,π].

    Array limits must be expressed as multiples of π.

    Parameters
    ----------
    x : np.ndarray
        The array to be modded.
    lim : list
        The mod limits.

    Returns
    -------
    2-tuple containing an np.ndarray and a list
        The modded array and list containing the limits.

    """
    if (lim != [0, 2 * np.pi]) and (lim != [-np.pi, np.pi]):
        logger.info("[mod 2π]Defaulting to [-π, π].")
        return _pi_mod(x, lim=[-np.pi, np.pi])

    if lim == [0, 2 * np.pi]:
        x_plot = np.mod(x, 2 * np.pi)
    elif lim == [-np.pi, np.pi]:
        x_plot = np.mod(x, 2 * np.pi)
        x_plot = x_plot - 2 * np.pi * (x_plot > np.pi)
    return x_plot, lim


def _auto_yspan(array: np.ndarray, psi_wall: float):
    """Calculates the plot span of a bounded array to be plotted.

    The "zoom out" and "hardylim" factors of the plot span can
    be tweaked in the plot_parameters.py under the name
    "auto_yaxis_zoom".

    Parameters
    ----------
    x : np.ndarray
        The array to be plotted.
    psi_wall : float
        The tokamak's psi_wall. Needed to
        calculate the correct hard-y limit.

    Returns
    -------
    2tuple
        2tuple containing the plot limits.

    """
    config = AutoYspan()

    zoomout = config.zoomout
    hardylim = config.hardylim * psi_wall

    xmin = array.min()
    xmax = array.max()

    diff = xmax - xmin
    mid = (xmin + xmax) / 2

    lower = max(0, mid - zoomout * diff)
    higher = min(mid + zoomout * diff, hardylim)

    xspan = (lower, higher)

    return xspan
