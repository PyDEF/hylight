"""
FIXME This does not work properly just now.
"""
import numpy as np
import scipy.fft, scipy.integrate

from .loader import load_phonons, load_poscar
from .constants import h_si, two_pi, eV_in_J, THz_in_meV, hbar_si, atomic_mass


def spectra(
    outcar,
    poscar_gs,
    poscar_es,
    zpl,
    sigma=None,
    resolution_e=1e-4,
    e_max=None,
    bias=0,
):
    """
    outcar, poscar_es, poscar_gs are path
    zpl is in eV
    sigma is in s-1 ?
    resolution_e in eV
    e_max in eV
    """

    if e_max is None:
        e_max = zpl * 2.5
    phonons, _, _ = load_phonons(outcar)
    delta_R = compute_delta_R(poscar_gs, poscar_es)

    return compute_spectra(
        phonons,
        delta_R,
        zpl,
        sigma,
        resolution_e,
        e_max,
        bias=bias,
    )


def compute_spectra(
    phonons, delta_R_tot, zpl, sigma, resolution_e, e_max, bias=0, window_fn=np.hamming, pre_convolve=None, use_q=False
):
    """
    zpl in eV
    delta_R_tot in A
    sigma in s-1
    resolution_e in eV
    e_max in eV
    bias in eV
    """

    bias_si = bias * eV_in_J

    sample_rate = e_max * eV_in_J / h_si

    resolution_t = 1 / sample_rate

    N = int(e_max / resolution_e)

    t = np.arange((-N) // 2 + 1, (N) // 2 + 1) * resolution_t

    # array of mode specific HR factors
    hrs = get_HR_factors(phonons, delta_R_tot * 1e-10, use_q=use_q)
    S = np.sum(hrs)

    # array of mode specific pulsations/radial frequencies
    energies = get_energies(phonons)

    freqs = energies / h_si * np.array(energies >= bias_si, dtype=float)

    s_t = get_s_t_raw(t, freqs, hrs)

    if pre_convolve is not None:
        sigma_s = h_si / (pre_convolve * eV_in_J)
        g = gaussian(t, sigma_s)
        s_t *= g / np.max(g)

    exp_s_t = np.exp(s_t)

    if sigma is None:
        line_shape = np.ones(t.shape, dtype=complex)
    elif sigma < 0:
        line_shape = np.array(np.exp(sigma * np.abs(t)), dtype=complex)
    else:
        line_shape = np.array(gaussian(t, 4.0 / sigma), dtype=complex)

    g_t = exp_s_t * np.exp(1.0j * two_pi * t * zpl * eV_in_J / h_si) * np.exp(-S)

    a_t = window(g_t * line_shape, fn=window_fn)

    e = np.arange(0, N) * resolution_e
    A = scipy.fft.fft(a_t)

    I = e ** 3 * A

    return e, I


def get_s_t_raw(t, freqs, hrs):
    # Fourier transform of individual S_i \delta {(\nu - \nu_i)}
    s_i_t = hrs.reshape((1, -1)) * np.exp(
        -1.0j * two_pi * freqs.reshape((1, -1)) * t.reshape((-1, 1))
    )

    # sum over the modes:
    return np.sum(s_i_t, axis=1)


def gaussian(e, sigma):
    return np.exp(-(e ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(two_pi))


def get_HR_factors(phonons, delta_R_tot, use_q=False):
    return np.array([ph.huang_rhys(delta_R_tot, use_q=use_q) for ph in phonons])


def get_energies(phonons):
    """Return an array of mode energies in SI"""
    return np.array([ph.energy for ph in phonons])


def compute_delta_R(poscar_gs, poscar_es):
    """Return $\\Delta R$ in A."""
    return load_poscar(poscar_gs) - load_poscar(poscar_es)


def rect(n):
    return np.ones((n,))


def window(data, fn=np.hanning):
    """Apply a windowing function to the data.
    Use hylight.multi_phonons.rect for as a dummy window.
    """
    n = len(data)
    return data * fn(n)


def stick_smooth_spectra(phonons, delta_R, height, n_points):
    """
    delta_R in A
    """
    ph_e_meV = [
        p.energy * 1000 / eV_in_J
        for p in phonons
    ]

    mi = min(ph_e_meV)
    ma = max(ph_e_meV)

    e_meV = np.linspace(mi, ma, n_points)

    w = 2 * (ma - mi) / n_points

    fc_spec = np.zeros(e_meV.shape)
    fc_sticks = np.zeros(e_meV.shape)

    for i, (e, hr) in enumerate(zip(ph_e_meV, get_HR_factors(phonons, delta_R * 1e-10))):
        h = height(hr, e)
        g_thin = gaussian(e_meV - e, w)
        g_fat = gaussian(e_meV - e, 1)
        fc_sticks += h * g_thin
        fc_spec += h * g_fat

    fc_sticks /= np.max(fc_sticks)
    fc_spec /= np.max(fc_spec)

    return e_meV, fc_spec, fc_sticks


def fc_spectra(phonons, delta_R, n_points=5000):
    return stick_smooth_spectra(phonons, delta_R, lambda hr, e: hr * e, n_points)


def hr_spectra(phonons, delta_R, n_points=5000):
    return stick_smooth_spectra(phonons, delta_R, lambda hr, _e: hr, n_points)
