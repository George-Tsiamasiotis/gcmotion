from pint import Quantity
from pprint import pprint


def pprint_dict(d: dict, everything: bool = False, units: str = ""):
    """Prints specific attributes of a particle.

    Only meant to be used internally by
    :py:meth:`~gcmotion.classes.particle.Particle.quantities()`.

    Parameters
    ----------
    d : dict
        The dictionary containing the attributes (from vars(self)).
    everything : bool, optional
        Whether or not to print *all* particle's *attributes*, and not just
        Quantities. Ignores "which" arguement if True. Defaults to False.
    units : str, optional
        The units of the attributes to be printed.
    """
    # Add everything, print and return
    if everything:
        pprint(
            [
                f"{key} = {value:.4g~P}"
                for key, value in d.items()
                if isinstance(value, Quantity)
            ]
            + [
                f"{key} = {value}"
                for key, value in d.items()
                if not isinstance(value, Quantity)
                and not isinstance(value, dict)
            ]
        )
        return

    res = []

    if units in ["NU", "SI"]:  # Add only one of the unit systems
        for key, value in d.items():
            if isinstance(value, Quantity):
                if units == "NU" and key[-2:] == "NU":
                    res.append(f"{key} = {value:.4g~P}")
                elif units == "SI" and key[-2:] != "NU":
                    res.append(f"{key} = {value:.4g~P}")
    else:
        for key, value in d.items():
            if isinstance(value, Quantity):
                res.append(f"{key} = {value:.4g~P}")

    pprint(res)
