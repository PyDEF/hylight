import logging

import numpy as np

from .constants import pi, kb_eV


def compute_spectra(
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
    sp = np.zeros(e.shape)

    if hard_osc:
        stokes_shift = fc_shift_e + fc_shift_g
        S = 0.5 * stokes_shift / e_phonon_g
        sig = sigma(T, S, e_phonon_g)
        khi_e_khi_g_squared = np.exp(-S)
    else:
        e_phonon_e = e_phonon_g * np.sqrt(fc_shift_e / fc_shift_g)
        S_abs = fc_shift_e / e_phonon_e
        S_em = fc_shift_g / e_phonon_g
        sig = sigma_soft(T, S_abs, S_em, e_phonon_g, e_phonon_e)
        S = S_em
        khi_e_khi_g_squared = np.exp(-S)

    details = []

    n, r = 0, 1

    while r > 1e-8 and n < 100:
        c = (
            e ** 3
            * khi_e_khi_g_squared
            * gaussian(e_zpl - n * e_phonon_g - e, sig)
        )
        details.append(c)
        sp += c
        r = np.max(c) / np.max(sp)
        n += 1
        khi_e_khi_g_squared *= S / n

    logging.info(f"went up to {n = }")

    return (
        e,
        sp / np.max(sp),
        [[c / np.max(sp) for c in details], 1 if hard_osc else e_phonon_e / e_phonon_g],
    )


def gaussian(e, sigma):
    return np.exp(-(e ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * pi))


def sigma(T, S, e_phonon):
    return e_phonon * np.sqrt(S) / np.sqrt(np.tanh(e_phonon / (kb_eV * T)))


def sigma_soft(T, S_abs, S_em, e_phonon_g, e_phonon_e):
    return (
        e_phonon_g
        * S_em
        / np.sqrt(S_abs)
        / np.sqrt(np.tanh(e_phonon_e / (2 * kb_eV * T)))
    )


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
