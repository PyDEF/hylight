"""A collection of useful physics constants.
"""
from __future__ import annotations

h_si: float = 6.62607015e-34  # SI (J.s)
c_si: float = 2.99792e8  # SI (m.s-1)
h_c: float = h_si * c_si  # SI (J.m)
eV_in_J: float = 1.602176634e-19
two_pi: float = 6.283185307179586
pi: float = 3.141592653589793
hbar_si: float = h_si / two_pi  # J.s.rad-1
hbar: float = h_si / two_pi / eV_in_J  # ev.s.rad-1

atomic_mass: float = 1.67e-27  # SI

THz_in_meV: float = 4.135667696923859

cm1_in_J: float = 100 * h_c

kb: float = 1.38064852e-23  # J.K-1 Boltzmann constant

kb_eV: float = kb / eV_in_J  # eV.K-1

sigma_to_fwhm: float = 2.3548200450309493  # 2 * sqrt(2 * log(2))


electronegativity: dict[str, float] = {
    # Pauling electronegativity, or 10.0 for species where no data is available
    "H": 2.20,
    "He": 10.0,
    "Li": 0.98,
    "Be": 1.57,
    "B": 2.04,
    "C": 2.55,
    "N": 3.04,
    "O": 3.44,
    "F": 3.98,
    "Ne": 10.0,
    "Na": 0.93,
    "Mg": 1.31,
    "Al": 1.61,
    "Si": 1.90,
    "P": 2.19,
    "S": 2.58,
    "Cl": 3.16,
    "Ar": 10.0,
    "K": 0.82,
    "Ca": 1.00,
    "Sc": 1.36,
    "Ti": 1.54,
    "V": 1.63,
    "Cr": 1.66,
    "Mn": 1.55,
    "Fe": 1.83,
    "Co": 1.88,
    "Ni": 1.91,
    "Cu": 1.90,
    "Zn": 1.65,
    "Ga": 1.81,
    "Ge": 2.01,
    "As": 2.18,
    "Se": 2.55,
    "Br": 2.96,
    "Kr": 3.00,
    "Rb": 0.82,
    "Sr": 0.95,
    "Y": 1.22,
    "Zr": 1.33,
    "Nb": 1.6,
    "Mo": 2.16,
    "Tc": 1.9,
    "Ru": 2.2,
    "Rh": 2.28,
    "Pd": 2.20,
    "Ag": 1.93,
    "Cd": 1.69,
    "In": 1.78,
    "Sn": 1.96,
    "Sb": 2.05,
    "Te": 2.1,
    "I": 2.66,
    "Xe": 2.6,
    "Cs": 0.79,
    "Ba": 0.89,
    "La": 1.10,
    "Ce": 1.12,
    "Pr": 1.13,
    "Nd": 1.14,
    "Pm": 1.13,
    "Sm": 1.17,
    "Eu": 1.2,
    "Gd": 1.2,
    "Tb": 1.22,
    "Dy": 1.23,
    "Ho": 1.24,
    "Er": 1.24,
    "Tm": 1.25,
    "Yb": 1.1,
    "Lu": 1.27,
    "Hf": 1.3,
    "Ta": 1.5,
    "W": 2.36,
    "Re": 1.9,
    "Os": 2.2,
    "Ir": 2.2,
    "Pt": 2.28,
    "Au": 2.54,
    "Hg": 2.00,
    "Tl": 1.62,
    "Pb": 2.33,
    "Bi": 2.02,
    "Po": 2.0,
    "At": 2.2,
    "Rn": 10.0,
    "Fr": 0.7,
    "Ra": 0.89,
    "Ac": 1.1,
    "Th": 1.3,
    "Pa": 1.5,
    "U": 1.38,
    "Np": 1.36,
    "Pu": 1.28,
    "Am": 1.3,
    "Cm": 1.3,
    "Bk": 1.3,
    "Cf": 1.3,
    "Es": 1.3,
    "Fm": 1.3,
    "Md": 1.3,
    "No": 1.3,
    "Lr": 10.0,
    "Rf": 10.0,
    "Db": 10.0,
    "Sg": 10.0,
    "Bh": 10.0,
    "Hs": 10.0,
    "Mt": 10.0,
    "Ds": 10.0,
    "Rg": 10.0,
    "Cn": 10.0,
    "Nh": 10.0,
    "Fl": 10.0,
    "Mc": 10.0,
    "Lv": 10.0,
    "Ts": 10.0,
    "Og": 10.0,
}
