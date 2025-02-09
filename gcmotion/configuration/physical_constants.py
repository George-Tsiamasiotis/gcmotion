from dataclasses import dataclass


@dataclass(frozen=True)
class PhysicalConstants:

    e_M: float = 0.0005446623  # Units of proton mass
    e_Z: int = -1  # Charge
    e_name: str = "Electron"  # Formal name, for display only

    p_M: float = 1
    p_Z: int = +1
    p_name: str = "Proton"

    d_M: float = 2
    d_Z: int = +1
    d_name: str = "Deuterium Ion"

    t_M: float = 3
    t_Z: int = +1
    t_name: str = "Tritium Ion"

    he3_M: float = 3
    he3_Z: int = +2
    he3_name: str = "He3 Ion"

    he4_M: float = 4
    he4_Z: int = +2
    he4_name: str = "He4 Ion"
