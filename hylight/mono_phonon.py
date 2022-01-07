import matplotlib.pyplot as plt
import numpy as np

from .loader import load_phonons, load_poscar
from .constants import h_si, hbar_si, two_pi, pi, eV_in_J, THz_in_meV, atomic_mass, kb_eV

from pydef.core.vasp import Poscar


def compute_spectrum(e_zpl, T, fc_shift_g, fc_shift_e, e_phonon_g, hard_osc=False, n_points=5000, e_min=0, e_max=5):
    e = np.linspace(e_min, e_max, n_points)
    sp = np.zeros(e.shape)

    if hard_osc:
        stokes_shift = fc_shift_e + fc_shift_g
        S = 0.5 * stokes_shift / e_phonon_g
        sig = sigma(T, S, e_phonon_g)
        khi_e_khi_g_squared = np.exp(-S)
    elif True:
        e_phonon_e = e_phonon_g * np.sqrt(fc_shift_e / fc_shift_g)
        S_abs = fc_shift_e / e_phonon_e
        S_em = fc_shift_g / e_phonon_g
        sig = sigma_soft(T, S_abs, S_em, e_phonon_g, e_phonon_e)
        S = S_em
        khi_e_khi_g_squared = np.exp(-S)

    details = []

    n, r = 0, 1

    while r > 1e-8 and n < 100:
        # FIXME Here I am not sure wether the right factor is e**3 or (e_zpl - n * e_phonon_g)**3
        # The answer depends on wether the factor should be applied before or
        # after broadening of the diracs
        c = (e_zpl - n * e_phonon_g)**3 * khi_e_khi_g_squared * gaussian(e_zpl - n * e_phonon_g - e, sig)
        details.append(c)
        sp += c
        r = np.max(c) / np.max(sp)
        n += 1
        khi_e_khi_g_squared *= S / n

    print(f"went up to {n = }")

    return e, sp / np.max(sp), [[c / np.max(sp) for c in details], 1 if hard_osc else e_phonon_e / e_phonon_g]


def gaussian(e, sigma):
    return np.exp(-(e ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * pi))


def sigma(T, S, e_phonon):
    return e_phonon * np.sqrt(S) / np.sqrt(np.tanh(e_phonon / (kb_eV * T)))


def sigma_soft(T, S_abs, S_em, e_phonon_g, e_phonon_e):
    return e_phonon_g * S_em / np.sqrt(S_abs) / np.sqrt(np.tanh(e_phonon_e / (2 * kb_eV * T)))


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


def phonon_composition(phonons, gs_path, es_path, n_points=5000):
    p_es = Poscar.from_file(es_path)
    p_gs = Poscar.from_file(gs_path)

    delta_R = p_es.raw - p_gs.raw

    freqs = [
        p.energy * 1000 / eV_in_J
        for p in phonons
    ]

    mi = min(freqs)
    ma = max(freqs)

    print(mi, ma)

    f = np.linspace(mi, ma, n_points)
    s = np.zeros(f.shape)
    raw_s = np.zeros(f.shape)

    for i, p in enumerate(phonons):
        fp = p.energy / hbar_si  # e [J] -> fp [rad.s-1]
        alpha_i = np.sum(p.delta * delta_R)
        alpha_i_si = alpha_i * 1e-10  # [A] -> [m]
        mu_i = p.mass * atomic_mass  # [am] -> [kg]
        fc = 0.5 * mu_i * fp**2 * alpha_i_si**2 / eV_in_J
        s += fc * gaussian(f - p.energy * 1000 / eV_in_J, 1)

        g = gaussian(f - p.energy * 1000 / eV_in_J, 1e-3)
        m = np.max(g)
        if m > 1e-5:
            raw_s += fc * g / m

    fig = plt.figure()
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)
    # ax3 = plt.subplot(3, 1, 3)

    ax1.sharex(ax2)
    ax1.set_ylabel("FC shift (meV)")
    ax1.stackplot(f, 1000 * s, color="grey")
    ax1.plot(f, 1000 * raw_s, "k")

    ax2.set_ylabel("HR factor")
    ax2.stackplot(f, 1000 * s / f, color="grey")
    ax2.plot(f, 1000 * raw_s / f, "k")

    # raw_d = np.zeros(f.shape)

    # for i, p in enumerate(phonons):
    #     alpha_i = abs(np.sum(p.delta * delta_R) / np.linalg.norm(delta_R))
    #     g = gaussian(f - p.energy * 1000 / eV_in_J, 5e-3)
    #     m = np.max(g)
    #     if m > 1e-5:
    #         raw_d += alpha_i * g / m

    # ax3.set_ylabel("Displacement contribution")
    # ax3.plot(f, raw_d, "k")

    return fig
