import numpy as np

from gcmotion.configuration.plot_parameters import energy_contour as config


def pi_mod(x, lim: list = [-np.pi, np.pi]) -> tuple[np.ndarray, list]:
    """Mods an array between [0,2π] or [-π,π].

    .. note::
        array limits must be expressed as multiples of
        ``np.pi``.

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
        print("x_lim must be either [0,2*np.pi] or [-np.pi,np.pi].")
        print("Defaulting to [-π, π].")
        return pi_mod(x, lim=[-np.pi, np.pi])

    if lim == [0, 2 * np.pi]:
        x_plot = np.mod(x, 2 * np.pi)
    elif lim == [-np.pi, np.pi]:
        x_plot = np.mod(x, 2 * np.pi)
        x_plot = x_plot - 2 * np.pi * (x_plot > np.pi)
    return x_plot, lim


def yspan(x: np.ndarray):
    """Calculates the plot span of a bounded array to be plotted,
    as well as the indexes of its total mimimum and maximum.

    The "zoom out" factor of the plot span can be tweaked in the
    plot_parameters.py under the name "auto_yaxis_zoom".

    Parameters
    ----------
    x : np.ndarray
        The array to be plotted.

    Returns
    -------
    2tuple of 2 2typles
        1st tuple contains the plot limits and 2nd tuple the
        minimum and maximum indeces

    """

    zoomout = config["auto_yaxis_zoom"]
    hardylim = config["hardylim"]

    xmin = x.min()
    xmax = x.max()
    minpos = x.argmin()
    maxpos = x.argmax()

    diff = xmax - xmin
    mid = (xmin + xmax) / 2

    lower = mid - zoomout * diff
    higher = min(mid + zoomout * diff, hardylim)

    xspan = (lower, higher)
    minmaxpos = (minpos, maxpos)

    return xspan, minmaxpos
