import numpy as np
from scipy import fft
from scipy.integrate import trapezoid as integrate

from .loader import load_phonons, load_poscar_latt
from .constants import two_pi, eV_in_J, h_si, pi, cm1_in_J, sigma_to_fwhm
from .mono_phonon import sigma_soft

from pydef.core.basic_functions import gen_translat

import logging

logger = logging.getLogger("hylight")


def spectra(
    outcar,
    poscar_gs,
    poscar_es,
    zpl,
    T,
    fc_shift_gs,
    fc_shift_es=None,
    e_max=None,
    resolution_e=1e-4,
    bias=0,
    load_phonons=load_phonons,
    pre_convolve=None,
    use_q=True,
):
    """
    :param path_vib: path to the vibration computation output file (by default an OUTCAR)
    :param path_struct_gs: path to the ground state relaxed structure file (by default a POSCAR)
    :param path_struct_es: path to the excited state relaxed structure file (by default a POSCAR)
    :param zpl: zero phonon line energy in eV
    :param T: temperature in K
    :param fc_shift_gs: Ground state/absorption Franck-Condon shift in eV
    :param fc_shift_es: (optional, equal to fc_shift_gs) Exited state/emmission Franck-Condon shift in eV
    :param e_max: max energy in eV (should be at least > 2*zpl)
    :param resolution_e: energy resolution in eV
    :param bias: (optional, 0) ignore low energy vibrations under bias in eV
    :param pre_convolve: (float, optional, None) if not None, standard deviation of the pre convolution gaussian
    :param load_phonons: a function that takes path_vib and produce a list of phonons. By default expect an OUTCAR.
    """

    if e_max is None:
        e_max = zpl * 2.5
    phonons, _, _ = load_phonons(outcar)
    delta_R = compute_delta_R(poscar_gs, poscar_es)

    if fc_shift_es is None:
        fc_shift_es = fc_shift_gs

    e, I = compute_spectra_soft(
        phonons,
        delta_R,
        zpl,
        T,
        fc_shift_gs,
        fc_shift_es,
        e_max,
        resolution_e,
        bias=bias,
        pre_convolve=pre_convolve,
        use_q=True,
    )

    return e, np.abs(I)


def plot_spectral_function(
    outcar, poscar_es, poscar_gs, load_phonons=load_phonons, use_q=True, use_cm1=False, disp=1
):
    from matplotlib import pyplot as plt

    phonons, _, _ = load_phonons(outcar)
    delta_R = compute_delta_R(poscar_gs, poscar_es)

    f, fc, dirac_fc = fc_spectra(phonons, delta_R, use_q=use_q, disp=disp)
    f, s, dirac_s = hr_spectra(phonons, delta_R, use_q=use_q, disp=disp)
    fc = s * f
    dirac_fc = dirac_s * f

    if use_cm1:
        f *= 1e-3 * eV_in_J / cm1_in_J

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.stackplot(f, s, color="grey")
    ax1.plot(f, dirac_s, color="black", lw=1)
    ax1.set_ylabel("$S(\\hbar\\omega)$ (A. U.)")

    ax2.stackplot(f, fc, color="grey")
    ax2.plot(f, dirac_fc, color="black", lw=1)
    ax2.set_ylabel("FC shift (meV)")
    if use_cm1:
        ax2.set_xlabel("Phonon frequency (cm$^{-1}$)")
    else:
        ax2.set_xlabel("Phonon energy (meV)")

    return fig


def compute_spectra_soft(
    phonons,
    delta_R,
    zpl,
    T,
    fc_shift_gs,
    fc_shift_es,
    e_max,
    resolution_e,
    bias=0,
    window_fn=np.hamming,
    pre_convolve=None,
    use_q=True,
):
    """
    :param phonons: list of modes
    :param delta_R: displacement in A
    :param zpl: zero phonon line energy in eV
    :param T: temperature in K
    :param fc_shift_gs: Ground state/absorption Franck-Condon shift in eV
    :param fc_shift_es: Exceited state/emmission Franck-Condon shift in eV
    :param e_max: max energy in eV (should be at least > 2*zpl)
    :param resolution_e: energy resolution in eV
    :param bias: (optional, 0) ignore low energy vibrations under bias in eV
    :param window_fn: (optional, np.hamming) windowing function in the form provided by numpy (see numpy.hamming)
    :param pre_convolve: (float, optional, None) if not None, standard deviation of the pre convolution gaussian
    :param use_q: (optional, True) if True use the DeltaQ when computing the HR factor, else use DeltaR
    """

    hrs = np.array(
        [
            ph.huang_rhys(delta_R * 1e-10, use_q=use_q)
            for ph in phonons
            if ph.energy >= bias * eV_in_J
        ]
    )
    fcs = (
        np.array(
            [
                ph.huang_rhys(delta_R * 1e-10, use_q=use_q) * ph.energy
                for ph in phonons
                if ph.energy >= bias * eV_in_J
            ]
        )
        / eV_in_J
    )

    S_abs = np.sum(hrs)

    # scale S_em according to the quotient of the FC shifts
    S_em = np.sum(hrs) / np.sqrt(fc_shift_es / fc_shift_gs)

    e_phonon_eff = np.sum(fcs) / S_abs

    # scale e_phonon_eff_e according to the quotient of the FC shifts
    e_phonon_eff_e = e_phonon_eff * np.sqrt(fc_shift_es / fc_shift_gs)

    logger.info(
        f"omega_gs = {e_phonon_eff * 1000} meV {e_phonon_eff * eV_in_J / cm1_in_J} cm-1"
    )
    logger.info(
        f"omega_es = {e_phonon_eff_e * 1000} meV {e_phonon_eff_e * eV_in_J / cm1_in_J} cm-1"
    )
    logger.info(f"S_abs = {S_abs}")
    logger.info(f"S_em = {S_em}")

    sig = sigma_soft(T, S_abs, S_em, e_phonon_eff, e_phonon_eff_e)

    fwhm = sig * sigma_to_fwhm
    logger.info(f"FWHM {fwhm * 1000} meV")

    return compute_spectra(
        phonons,
        delta_R,
        zpl,
        fwhm,
        e_max,
        resolution_e,
        bias=bias,
        window_fn=window_fn,
        pre_convolve=pre_convolve,
        use_q=use_q,
    )


def compute_spectra(
    phonons,
    delta_R,
    zpl,
    fwhm,
    e_max,
    resolution_e,
    bias=0,
    window_fn=np.hamming,
    pre_convolve=None,
    use_q=True,
):
    """
    :param phonons: list of modes
    :param delta_R: displacement in A
    :param zpl: zero phonon line energy in eV
    :param fwhm: zpl lineshape full width at half maximum in eV or None
      if fwhm is None: the raw spectra is not convoluted with a line shape
      if fwhm < 0: the spectra is convoluted with a lorentizan line shape
      if fwhm > 0: the spectra is convoluted with a gaussian line shape
    :param e_max: max energy in eV (should be at least > 2*zpl)
    :param resolution_e: energy resolution in eV
    :param bias: (optional, 0) ignore low energy vibrations under bias in eV
    :param window_fn: (optional, np.hamming) windowing function in the form provided by numpy (see numpy.hamming)
    :param pre_convolve: (float, optional, None) if not None, standard deviation of the pre convolution gaussian
    :param use_q: (optional, True) if True use the DeltaQ whane computing the HR factor, else use DeltaR
    """

    bias_si = bias * eV_in_J

    if fwhm is None:
        sigma_si = None
    else:
        sigma_si = fwhm * eV_in_J / sigma_to_fwhm

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

    s_t = _get_s_t_raw(t, freqs, hrs)

    if pre_convolve is not None:
        sigma_freq = pre_convolve * eV_in_J / h_si / sigma_to_fwhm
        g = _gaussian(t, 1 / (two_pi * sigma_freq))
        if np.max(g) > 0:
            s_t *= g / np.max(g)

    exp_s_t = np.exp(s_t - S)

    g_t = exp_s_t * np.exp(1.0j * two_pi * t * zpl * eV_in_J / h_si)

    if sigma_si is None:
        line_shape = np.ones(t.shape, dtype=complex)
    elif sigma_si <= 0:
        sigma_freq = sigma_si / h_si
        line_shape = np.array(np.exp(pi * sigma_freq * np.abs(t)), dtype=complex)
    else:
        sigma_freq = sigma_si / h_si
        line_shape = np.sqrt(2) * np.array(
            _gaussian(t, 1 / (two_pi * sigma_freq)), dtype=complex
        )

    a_t = _window(g_t * line_shape, fn=window_fn)

    e = np.arange(0, N) * resolution_e
    A = fft.fft(a_t)

    I = e**3 * A

    return e, I


def get_HR_factors(phonons, delta_R_tot, use_q=True):
    """
    delta_R_tot in SI
    """
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


def _get_s_t_raw(t, freqs, hrs):
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


def _gaussian(e, sigma):
    return np.exp(-(e**2) / (2 * sigma**2)) / (sigma * np.sqrt(two_pi))


def _window(data, fn=np.hamming):
    """Apply a windowing function to the data.
    Use hylight.multi_phonons.rect for as a dummy window.
    """
    n = len(data)
    return data * fn(n)


def fc_spectra(phonons, delta_R, n_points=5000, use_q=True, disp=1):
    f, s, dirac_s = _stick_smooth_spectra(
        phonons, delta_R, lambda hr, e: hr * e, n_points, use_q, disp=disp
    )

    return f, f * s, f * dirac_s


def hr_spectra(phonons, delta_R, n_points=5000, use_q=True, disp=1):
    return _stick_smooth_spectra(phonons, delta_R, lambda hr, _e: hr, n_points, use_q, disp=disp)


def _stick_smooth_spectra(phonons, delta_R, height, n_points, use_q, disp=1):
    """
    delta_R in A
    """
    ph_e_meV = [p.energy * 1000 / eV_in_J for p in phonons]

    mi = min(ph_e_meV)
    ma = max(ph_e_meV)

    e_meV = np.linspace(mi, ma + 0.2 * (ma - mi), n_points)

    w = 2 * (ma - mi) / n_points

    fc_spec = np.zeros(e_meV.shape)
    fc_sticks = np.zeros(e_meV.shape)

    hrs = get_HR_factors(phonons, delta_R * 1e-10, use_q=use_q)

    for e, hr in zip(ph_e_meV, hrs):
        h = height(hr, e)
        g_thin = _gaussian(e_meV - e, w)
        g_fat = _gaussian(e_meV - e, disp)
        fc_sticks += h * g_thin / np.max(g_thin)  # g_thin should be h high
        fc_spec += h * g_fat  # g_fat has a g area

    return e_meV, fc_spec, fc_sticks


def rect(n):
    """A dummy windowing function that works like numpy.hamming, but as no effect on data."""
    return np.ones((n,))
