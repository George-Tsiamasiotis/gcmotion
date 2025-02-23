"""
Sets up the `pint <https://pint.readthedocs.io/en/stable/>`_ configuration.
"""

from pint import UnitRegistry, set_application_registry
from gcmotion.configuration.physical_constants import PhysicalConstants

from gcmotion.utils.logger_setup import logger
from gcmotion.utils.decorators import _calls_counter


@_calls_counter
def QuantityConstructor(
    R: float,
    B0: float,
    species: str,
    a: float = None,
    _psi_wallNU: float = None,
):
    r"""Extends pint's default Quantity Constructor to include NU units and
    some extra SI unit aliases.

    Parameters
    ----------
    R : float
        The tokamak's major radius **in [m]**.
    a : float
        The tokamak's minor radius **in [m]**. Only used to create "psi_wall"
        and "NUpsi_wall" as units of magnetic flux, so it is possible to setup
        initial :math:`\psi_0` conditions with respect to the wall, instead of
        guessing. In reconstructed equilibria, a is defined *through* psi_wall
        and has a purely decorative purpose, and is not used in any
        calculations.
    B0 : float
        The magnetic field strength on the magnetic axis **in [T]**.
    species : str
        The particle species.

    Returns
    -------
    pint.Quantity
        The updated pint Quantity Constructor

    """

    if QuantityConstructor.calls == 1:
        logger.info("Defining Quantity Constructor")
    else:
        logger.warning("Redefining Quantity Constructor...")

    a_str = "None" if a is None else f"{a:.4g}"
    _psi_wallNU_str = "None" if _psi_wallNU is None else f"{_psi_wallNU:.4g}"
    logger.debug(
        "\tConstructor parameters: "
        f"B0={B0:.4g}, R={R:.4g}, a={a_str}, species={species}, "
        f"_psi_wallNU={_psi_wallNU_str}"
    )

    # NOTE: we can set "on_redefinition='ignore'" here in case the consructor
    # is need somewhere and its not accessible without redifining, but lets try
    # to avoid that, its bad practice. The constructor should be a singleton.
    ureg = UnitRegistry(case_sensitive=False)

    # Create the UnitRegistry
    ureg.setup_matplotlib()

    # Additional SI quantites (= aliases, for display only)
    ureg.define("Magnetic_flux       = Tesla * m^2   = Tm^2 = mf")
    ureg.define("Magnetic_moment     = Ampere * m^2  = keV/T")
    ureg.define("Plasma_current      = Tesla * m     = Tm")
    ureg.define("Canonical_momentum  = Joule * seconds = Js = canmom")

    # Base NU units
    mp = 1.672621923e-27
    qp = 1.602176634e-19

    ureg.define(f"Proton_mass   = {mp} kilogram = NUkilogram")
    ureg.define(f"Proton_charge = {qp} coulomb = NUCoulomb")

    M = getattr(PhysicalConstants, species.lower() + "_M")
    Z = getattr(PhysicalConstants, species.lower() + "_Z")

    w0 = (Z / M) * qp / mp * B0  # s^-1
    E0 = mp * w0**2 * R**2  # Joule

    ureg.define(f"NUsecond = {1/w0} second")
    ureg.define(f"NUw0     = {w0} hz")
    ureg.define(f"NUmeter  = {R} meter")
    ureg.define(f"NUJoule  = {E0} Joule")
    ureg.define(f"NUkeV    = {qp} NUJoule")
    ureg.define(f"NUTesla  = {M/Z} Proton_mass * NUw0 / Proton_charge ")

    # Additional NU quantities
    ureg.define("NUvelocity = NUmeter * NUw0")
    ureg.define("NUPlasma_current = NUTesla * NUmeter = NUpc")
    ureg.define("NUVolts = NUJoule / Proton_charge = NUV")
    ureg.define("NUVolts_per_NUmeter = NUVolts / NUmeter")
    ureg.define(
        "NUCanonical_momentum = Proton_mass * NUw0 * NUmeters^2 = NUcanmom"
    )
    ureg.define(
        "NUMagnetic_moment = Proton_charge / NUsecond * NUmeter^2 = NUmu"
    )
    ureg.define(
        "NUMagnetic_flux = "
        "Proton_mass * NUw0 * NUmeter^2 / Proton_charge = NUmf"
    )

    # Also define psi_wall as a unit of Magnetic_flux, to assign psi
    # initial values with respect to it
    if a is not None:
        _psi_wall = B0 * a**2 / 2
        psi_wall = ureg.Quantity(_psi_wall, "Magnetic_flux")
        ureg.define(f"psi_wall = {psi_wall}")
        ureg.define(f"NUpsi_wall= {psi_wall.to("NUmagnetic_flux")}")
    else:
        psi_wallNU = ureg.Quantity(_psi_wallNU, "NUMagnetic_flux")
        psi_wall = psi_wallNU.to("Magnetic_flux")
        ureg.define(f"psi_wall = {psi_wall.m} Magnetic_flux")
        ureg.define(f"NUpsi_wall = {psi_wallNU.m} NUMagnetic_flux")

    # This hopefully resets the UnitRegistry, which might cause wrong unit
    # definition if this function is called many times with the different
    # parameters inside the same session
    set_application_registry(ureg)
    return ureg.Quantity
