"A collection of useful physics constants."
# License
#     Copyright (C) 2023  PyDEF development team
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.
from __future__ import annotations

h_si: float = 6.62607015e-34  # SI (J.s)
c_si: float = 2.99792458e8  # SI (m.s-1)
h_c: float = h_si * c_si  # SI (J.m)
eV_in_J: float = 1.602176634e-19
two_pi: float = 6.283185307179586
pi: float = 3.141592653589793
hbar_si: float = h_si / two_pi  # J.s.rad-1
hbar: float = h_si / two_pi / eV_in_J  # eV.s.rad-1
electron_mass: float = 9.1093837015e-31  # kg
e_si: float = 1.60217663e-19  # C

atomic_mass: float = 1.660539066605e-27  # kg

THz_in_meV: float = 4.135667696923859

cm1_in_J: float = 100 * h_c

kb: float = 1.38064852e-23  # J.K-1 Boltzmann constant

kb_eV: float = kb / eV_in_J  # eV.K-1

eV1_in_nm: float = h_c * 1e9 / eV_in_J

sigma_to_fwhm: float = 2.3548200450309493  # 2 * sqrt(2 * log(2))

inv_four_pi_eps: float = 8.987_551_784e9  # SI
epsilon_0: float = 8.854_187_82e-12  # SI

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
}

masses = {
    "H": 1.008,
    "He": 4.002602,
    "Li": 6.94,
    "Be": 9.0121831,
    "B": 10.81,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998403163,
    "Ne": 20.1797,
    "Na": 22.98976928,
    "Mg": 24.305,
    "Al": 26.9815385,
    "Si": 28.085,
    "P": 30.973761998,
    "S": 32.06,
    "Cl": 35.45,
    "Ar": 39.948,
    "K": 39.0983,
    "Ca": 40.078,
    "Sc": 44.955908,
    "Ti": 47.867,
    "V": 50.9415,
    "Cr": 51.9961,
    "Mn": 54.938044,
    "Fe": 55.845,
    "Co": 58.933194,
    "Ni": 58.6934,
    "Cu": 63.546,
    "Zn": 65.38,
    "Ga": 69.723,
    "Ge": 72.63,
    "As": 74.921595,
    "Se": 78.971,
    "Br": 79.904,
    "Kr": 83.798,
    "Rb": 85.4678,
    "Sr": 87.62,
    "Y": 88.90584,
    "Zr": 91.224,
    "Nb": 92.90637,
    "Mo": 95.95,
    "Tc": 97.90721,
    "Ru": 101.07,
    "Rh": 102.9055,
    "Pd": 106.42,
    "Ag": 107.8682,
    "Cd": 112.414,
    "In": 114.818,
    "Sn": 118.71,
    "Sb": 121.76,
    "Te": 127.6,
    "I": 126.90447,
    "Xe": 131.293,
    "Cs": 132.90545196,
    "Ba": 137.327,
    "La": 138.90547,
    "Ce": 140.116,
    "Pr": 140.90766,
    "Nd": 144.242,
    "Pm": 144.91276,
    "Sm": 150.36,
    "Eu": 151.964,
    "Gd": 157.25,
    "Tb": 158.92535,
    "Dy": 162.5,
    "Ho": 164.93033,
    "Er": 167.259,
    "Tm": 168.93422,
    "Yb": 173.054,
    "Lu": 174.9668,
    "Hf": 178.49,
    "Ta": 180.94788,
    "W": 183.84,
    "Re": 186.207,
    "Os": 190.23,
    "Ir": 192.217,
    "Pt": 195.084,
    "Au": 196.966569,
    "Hg": 200.592,
    "Tl": 204.38,
    "Pb": 207.2,
    "Bi": 208.9804,
    "Po": 208.98243,
    "At": 209.98715,
    "Rn": 222.01758,
    "Fr": 223.01974,
    "Ra": 226.02541,
    "Ac": 227.02775,
    "Th": 232.0377,
    "Pa": 231.03588,
    "U": 238.02891,
    "Np": 237.04817,
    "Pu": 244.06421,
    "Am": 243.06138,
    "Cm": 247.07035,
    "Bk": 247.07031,
    "Cf": 251.07959,
    "Es": 252.083,
    "Fm": 257.09511,
    "Md": 258.09843,
    "No": 259.101,
    "Lr": 262.11,
}
