import numpy as np
from scipy import fft
from scipy.integrate import trapezoid as integrate

from .loader import load_phonons, load_poscar_latt
from .constants import two_pi, eV_in_J, h_si

from pydef.core.basic_functions import gen_translat


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
    sigma is in eV
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
    phonons,
    delta_R,
    zpl,
    sigma,
    resolution_e,
    e_max,
    bias=0,
    window_fn=np.hamming,
    pre_convolve=None,
    use_q=False,
):
    """
    :param zpl: zero phonon line energy in eV
    :param delta_R: displacement in A
    :param sigma: zpl line width in s-1
    :param resolution_e: energy resolution in eV
    :param e_max: max energy in eV (should be at least > 2*zpl)
    :param bias: ignore low energy vibrations under bias in eV
    """

    bias_si = bias * eV_in_J

    sample_rate = e_max * eV_in_J / h_si

    resolution_t = 1 / sample_rate

    N = int(e_max / resolution_e)

    t = np.arange((-N) // 2 + 1, (N) // 2 + 1) * resolution_t

    # array of mode specific HR factors
    hrs = get_HR_factors(phonons, delta_R * 1e-10, use_q=use_q)
    S = np.sum(hrs)

    # array of mode specific pulsations/radial frequencies
    energies = get_energies(phonons)

    freqs = energies / h_si * np.array(energies >= bias_si, dtype=float)

    s_t = get_s_t_raw(t, freqs, hrs)

    if pre_convolve is not None:
        sigma_s = h_si / (pre_convolve * eV_in_J)
        g = gaussian(t, sigma_s)
        if np.max(g) > 0:
            s_t *= g / np.max(g)

    exp_s_t = np.exp(s_t - S)

    g_t = exp_s_t * np.exp(1.0j * two_pi * t * zpl * eV_in_J / h_si)

    if sigma is None:
        line_shape = np.ones(t.shape, dtype=complex)
        print("No line shape")
    elif sigma < 0:
        sigma_si = sigma * eV_in_J / h_si
        line_shape = np.array(np.exp(sigma_si * np.abs(t)), dtype=complex)
        print("Lorentzian")
    else:
        sigma_si = sigma * eV_in_J / h_si
        line_shape = np.array(gaussian(t, 4.0 / sigma_si), dtype=complex)

        print("Gaussian")

    a_t = window(g_t * line_shape, fn=window_fn)

    e = np.arange(0, N) * resolution_e
    A = fft.fft(a_t)

    I = e ** 3 * A

    return e, I


def get_s_t_raw(t, freqs, hrs):
    # Fourier transform of individual S_i \delta {(\nu - \nu_i)}
    try:
        # This can create a huge array if freqs is too big
        # but it let numpy handle everything so it is really fast
        s_i_t = hrs.reshape((1, -1)) * np.exp(
            -1.0j * two_pi * freqs.reshape((1, -1)) * t.reshape((-1, 1))
        )
    except MemoryError:
        # Slower less memory intensive solution
        s_t = np.zeros(t.shape, dtype=complex)
        for hr, fr in zip(hrs, freqs):
            s_t += hr * np.exp(-1.0j * two_pi * fr * t)
        return s_t
    else:
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

    pos1, lattice1 = load_poscar_latt(poscar_gs)
    pos2, lattice2 = load_poscar_latt(poscar_es)

    if np.linalg.norm(lattice1 - lattice2) > 1e-5:
        raise ValueError("Lattice parameters are not matching.")

    dp = pos1 - pos2
    d = np.array([dp - t for t in gen_translat(lattice1)])  # shape (27, n, 3)

    # norms is an array of distances between a point in p1 a translated point of p2
    norms = np.linalg.norm(d, axis=2)  # shape = (27, n)
    # permut[i] is the index of the translated point of p2 that
    # is closer to p1[i]
    best_translat = np.argmin(norms, axis=0)  # shape = (n,)

    n = d.shape[1]
    res = np.ndarray((n, 3))

    for i, k in enumerate(best_translat):
        res[i, :] = d[k, i, :]

    return res


def rect(n):
    return np.ones((n,))


def window(data, fn=np.hanning):
    """Apply a windowing function to the data.
    Use hylight.multi_phonons.rect for as a dummy window.
    """
    n = len(data)
    return data * fn(n)


def stick_smooth_spectra(phonons, delta_R, height, n_points, use_q):
    """
    delta_R in A
    """
    ph_e_meV = [p.energy * 1000 / eV_in_J for p in phonons]

    mi = min(ph_e_meV)
    ma = max(ph_e_meV)

    e_meV = np.linspace(mi, ma, n_points)

    w = 2 * (ma - mi) / n_points

    fc_spec = np.zeros(e_meV.shape)
    fc_sticks = np.zeros(e_meV.shape)

    max_hr = -1

    hrs = get_HR_factors(phonons, delta_R * 1e-10, use_q=use_q)
    S = np.sum(hrs)
    max_hr = np.max(hrs)

    for i, (e, hr) in enumerate(zip(ph_e_meV, hrs)):
        h = height(hr, e)
        g_thin = gaussian(e_meV - e, w)
        g_fat = gaussian(e_meV - e, 1)
        fc_sticks += h * g_thin
        fc_spec += h * g_fat

    fc_sticks *= max_hr / np.max(fc_sticks)
    fc_spec *= S / integrate(fc_spec, e_meV)

    return e_meV, fc_spec, fc_sticks


def fc_spectra(phonons, delta_R, n_points=5000, use_q=False):
    return stick_smooth_spectra(phonons, delta_R, lambda hr, e: hr * e, n_points, use_q)


def hr_spectra(phonons, delta_R, n_points=5000, use_q=False):
    return stick_smooth_spectra(phonons, delta_R, lambda hr, _e: hr, n_points, use_q)
