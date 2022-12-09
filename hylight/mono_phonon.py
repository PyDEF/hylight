import logging

import numpy as np

from .constants import kb_eV
from .utils import gaussian


logger = logging.getLogger("hylight")


def spectra(
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
    """Compute a spectra from a single vibrational mode and some energetic terms.

    :param e_zpl: energy of the zero phonon line, in eV
    :param T: temperature in K
    :param fc_shift_g: Franck-Condon shift of emmission in eV
    :param fc_shift_e: Franck-Condon shift of absobtion in eV
    :param e_phonon_g: Mode energy in ground state in eV
    :param hard_osc: boolean, (optional, False) use the hard oscillator mode:
      vibration mode has the same energy in GD and ES
    :param n_points: (optional, 5000) number of points in the spectra
    :param e_min: (optional, 0) energy lower bound for the spectra, in eV
    :param e_max: (optional, 5) energy higher bound for the spectra, in eV
    """

    e = np.linspace(e_min, e_max, n_points)

    if hard_osc:
        stokes_shift = fc_shift_e + fc_shift_g
        S = 0.5 * stokes_shift / e_phonon_g
        sig = sigma(T, S, e_phonon_g)
    else:
        e_phonon_e = e_phonon_g * np.sqrt(fc_shift_e / fc_shift_g)
        S_em = fc_shift_g / e_phonon_g
        S = S_em
        sig = sigma_soft(T, S_em, e_phonon_g, e_phonon_e)

    return compute_spectra(e, e_zpl, S, sig, e_phonon_e)


def compute_spectra(
    e,
    e_zpl,
    S,
    sig,
    e_phonon_g,
):
    """Compute a spectra from 1D model with experimental like inputs

    :param e: a numpy array of energies to compute the spectra at
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


def sigma(T, S, e_phonon):
    if T == 0.0:
        return e_phonon * np.sqrt(S)
    else:
        return e_phonon * np.sqrt(S / np.tanh(e_phonon / (kb_eV * T)))


def sigma_soft(T, S_em, e_phonon_g, e_phonon_e):
    if T == 0.0:
        coth = 1.0
    else:
        coth = 1.0 / np.tanh(e_phonon_e / (2 * kb_eV * T))

    return np.sqrt(S_em * e_phonon_g**3 / e_phonon_e * coth)


def huang_rhys(stokes_shift, e_phonon):
    return 0.5 * stokes_shift / e_phonon


def best_max(x, y, f=0.95):
    guess = np.max(y)

    sx = 0
    sy = 0
    c = 0

    for vx, vy in zip(x, y):
        if vy > f * guess:
            sy += vy
            sx += vx
            c += 1
    return sx / c, sy / c
