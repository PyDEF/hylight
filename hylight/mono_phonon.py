"Simulation of spectra in 1D model."
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
import logging

import numpy as np

from .constants import kb_eV
from .utils import gaussian


logger = logging.getLogger("hylight")


def spectrum(
    e_zpl,
    T,
    fc_shift_g,
    fc_shift_e,
    e_phonon_g,
    hard_osc=False,
    n_points=5000,
    e_min=0,
    e_max=5,
):
    """Compute a spectrum from a single vibrational mode and some energetic terms.

    :param e_zpl: energy of the zero phonon line, in eV
    :param T: temperature in K
    :param fc_shift_g: Franck-Condon shift of emmission in eV
    :param fc_shift_e: Franck-Condon shift of absobtion in eV
    :param e_phonon_g: Mode energy in ground state in eV
    :param hard_osc: boolean, use the hard oscillator mode:
      vibration mode has the same energy in GD and ES
    :param n_points: number of points in the spectrum
    :param e_min: energy lower bound for the spectrum, in eV
    :param e_max: energy higher bound for the spectrum, in eV
    """

    e = np.linspace(e_min, e_max, n_points)

    if hard_osc:
        stokes_shift = fc_shift_e + fc_shift_g
        S = 0.5 * stokes_shift / e_phonon_g
        sig = sigma(T, S, e_phonon_g, e_phonon_g)
    else:
        e_phonon_e = e_phonon_g * np.sqrt(fc_shift_e / fc_shift_g)
        S = fc_shift_g / e_phonon_g
        sig = sigma(T, S, e_phonon_g, e_phonon_e)

    return compute_spectrum(e, e_zpl, S, sig, e_phonon_e)


def compute_spectrum(
    e,
    e_zpl,
    S,
    sig,
    e_phonon_g,
):
    """Compute a spectrum from 1D model with experimental like inputs

    :param e: a numpy array of energies to compute the spectrum at
    :param e_zpl: energy of the zero phonon line, in eV
    :param S: the emission Huang-Rhys factor
    :param sig: the lineshape standard deviation
    :e_phonon_g: the ground state phonon energy
    """

    sp = np.zeros(e.shape)

    khi_e_khi_g_squared = np.exp(-S)

    details = []

    n, r = 0, 1

    while r > 1e-8 and n < 100:
        c = e**3 * khi_e_khi_g_squared * gaussian(e_zpl - n * e_phonon_g - e, sig)
        details.append(c)
        sp += c
        r = np.max(c) / np.max(sp)
        n += 1
        khi_e_khi_g_squared *= S / n

    logger.info(f"Summed up to {n - 1} replicas.")

    return (
        e,
        sp / np.max(sp),
        [[c / np.max(sp) for c in details]],
    )


def sigma(T, S_em, e_phonon_g, e_phonon_e):
    """Temperature dependant standard deviation of the lineshape.

    :param T: temperature in K
    :param S_em: emmission Huang-Rhys factor
    :param e_phonon_g: energy of the GS PES vibration (eV)
    :param e_phonon_e: energy of the ES PES vibration (eV)
    """
    coth = 1.0 / np.tanh(e_phonon_e / (2 * kb_eV * T)) if T > 0.0 else 1.0

    return np.sqrt(S_em * e_phonon_g**3 / e_phonon_e * coth)


def huang_rhys(stokes_shift, e_phonon):
    """Huang-Rhys factor from the Stokes shift and the phonon energy."""
    return 0.5 * stokes_shift / e_phonon
