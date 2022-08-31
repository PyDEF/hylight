import warnings
from enum import Enum
import numpy as np
from scipy import fft

from .loader import load_phonons, load_poscar_latt
from .constants import two_pi, eV_in_J, h_si, pi, cm1_in_J, sigma_to_fwhm, hbar_si
from .mono_phonon import sigma_soft
from .utils import gen_translat

import logging

logger = logging.getLogger("hylight")


class LineShape(Enum):
    GAUSSIAN = 0
    LORENTZIAN = 1
    NONE = 2


class OmegaEff(Enum):
    # FC_MEAN should be used with ExPES.ISO_SCALE
    # because it is associated with the idea that all the directions are
    # softened equally in the excited state
    FC_MEAN = 0
    # HR_MEAN means Omega_eff = d_FC,tot / S_tot
    HR_MEAN = 1
    # I don't now if this one makes any sense
    HR_RMS = 2
    # FC_RMS should be used with ExPES.SINGLE_ES_FREQ
    # because it makes sense when we get only one Omega_eff for the excited
    # state (this single effective frequency should be computed beforehand)
    FC_RMS = 3


class _ExPES(Enum):
    ISO_SCALE = 0
    SINGLE_ES_FREQ = 1
    FULL_ND = 2


class ExPES:
    ISO_SCALE: "ExPES"
    SINGLE_ES_FREQ: "ExPES"
    FULL_ND: "ExPES"

    def __init__(self, wm):
        self.wm = wm
        self.omega = None

    def __eq__(self, other):
        return isinstance(other, ExPES) and self.wm == other.wm

    def __call__(self, omega=None):
        new = ExPES(self.wm)
        new.omega = omega
        return new


ExPES.ISO_SCALE = ExPES(_ExPES.ISO_SCALE)
ExPES.SINGLE_ES_FREQ = ExPES(_ExPES.SINGLE_ES_FREQ)
ExPES.FULL_ND = ExPES(_ExPES.FULL_ND)


def spectra(
    outcar,
    poscar_gs,
    poscar_es,
    zpl,
    T,
    fc_shift_gs=None,
    fc_shift_es=None,
    e_max=None,
    resolution_e=1e-4,
    bias=0,
    load_phonons=load_phonons,
    pre_convolve=None,
    shape=LineShape.GAUSSIAN,
    omega_eff_type=OmegaEff.FC_MEAN,
    result_store=None,
    ex_pes=ExPES.ISO_SCALE,
    correct_zpe=False,
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
        shape=shape,
        omega_eff_type=omega_eff_type,
        result_store=result_store,
        ex_pes=ex_pes,
        correct_zpe=correct_zpe,
    )

    return e, np.abs(I)


def plot_spectral_function(
    outcar,
    poscar_es,
    poscar_gs,
    load_phonons=load_phonons,
    use_cm1=False,
    disp=1,
    mpl_params=None,
):
    from matplotlib import pyplot as plt

    phonons, _, _ = load_phonons(outcar)
    delta_R = compute_delta_R(poscar_gs, poscar_es)

    f, fc, dirac_fc = fc_spectra(phonons, delta_R, disp=disp)
    f, s, dirac_s = hr_spectra(phonons, delta_R, disp=disp)
    fc = s * f
    dirac_fc = dirac_s * f

    if use_cm1:
        f *= 1e-3 * eV_in_J / cm1_in_J

    s_stack = {"color": "grey"}
    fc_stack = {"color": "grey"}
    s_peaks = {"color": "black", "lw": 1}
    fc_peaks = {"color": "black", "lw": 1}

    if mpl_params:
        s_stack.update(mpl_params.get("S_stack", {}))
        fc_stack.update(mpl_params.get("FC_stack", {}))
        s_peaks.update(mpl_params.get("S_peaks", {}))
        fc_peaks.update(mpl_params.get("FC_peaks", {}))

    fig, (ax_fc, ax_s) = _, (_, ax_bottom) = plt.subplots(2, 1, sharex=True)
    ax_s.stackplot(f, s, **s_stack)
    ax_s.plot(f, dirac_s, **s_peaks)
    ax_s.set_ylabel("$S(\\hbar\\omega)$ (A. U.)")

    ax_fc.stackplot(f, fc, **fc_stack)
    ax_fc.plot(f, dirac_fc, **fc_peaks)
    ax_fc.set_ylabel("FC shift (meV)")

    if use_cm1:
        ax_bottom.set_xlabel("Wavenumber (cm$^{-1}$)")
    else:
        ax_bottom.set_xlabel("E (meV)")

    return fig, (ax_fc, ax_s)


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
    shape=LineShape.GAUSSIAN,
    omega_eff_type=OmegaEff.FC_MEAN,
    result_store=None,
    ex_pes=ExPES.ISO_SCALE,
    correct_zpe=False,
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
    """

    if result_store is None:
        result_store = {}


    bias_si = bias * eV_in_J

    hrs = get_HR_factors(phonons, delta_R * 1e-10, bias=bias_si)
    es = get_energies(phonons, bias=bias_si) / eV_in_J
    fcs = hrs * es

    S_em = np.sum(hrs)
    dfcg_vib = np.sum(fcs)

    if omega_eff_type == OmegaEff.FC_MEAN:
        e_phonon_eff = np.sum(fcs * es) / dfcg_vib
    elif omega_eff_type == OmegaEff.HR_MEAN:
        e_phonon_eff = dfcg_vib / S_em
    elif omega_eff_type == OmegaEff.HR_RMS:
        e_phonon_eff = np.sqrt(np.sum(fcs * es) / S_em)
    elif omega_eff_type == OmegaEff.FC_RMS:
        e_phonon_eff = np.sqrt(np.sum(fcs * es * es) / dfcg_vib)

    if ex_pes.omega is None:
        assert fc_shift_gs is not None and fc_shift_gs > 0
        assert fc_shift_es is not None
        alpha = np.sqrt(fc_shift_es / fc_shift_gs)
        e_phonon_eff_e = e_phonon_eff * alpha
        logger.info(f"d_fc^e,v = {dfcg_vib * alpha**2}")
        result_store["d_fc^e,v"] = dfcg_vib * alpha**2
        result_store["alpha"] = alpha
    else:
        e_phonon_eff_e = ex_pes.omega
        alpha = e_phonon_eff_e / e_phonon_eff
        logger.info(f"d_fc^e,v = {dfcg_vib * alpha**2}")
        result_store["d_fc^e,v"] = dfcg_vib * alpha**2
        result_store["alpha"] = alpha

    if ex_pes == ExPES.ISO_SCALE:
        sig = sigma_soft(T, S_em, e_phonon_eff, e_phonon_eff_e)
        sig0 = sigma_soft(0, S_em, e_phonon_eff, e_phonon_eff_e)
    elif ex_pes == ExPES.SINGLE_ES_FREQ:
        if ex_pes.omega is not None:
            e_phonon_eff_e = ex_pes.omega

        sig = sigme_hybrid(T, hrs, es, e_phonon_eff_e)
        sig0 = sigme_hybrid(0, hrs, es, e_phonon_eff_e)
    else:
        raise ValueError("Unexpected width model.")

    if correct_zpe:
        zpe_gs = 0.5 * np.sum([p.energy for p in phonons]) / eV_in_J
        if ex_pes == ExPES.ISO_SCALE:
            gamma = e_phonon_eff_e / e_phonon_eff
            zpe_es = zpe_gs * gamma
        elif ex_pes == ExPES.SINGLE_ES_FREQ:
            zpe_es = 0.5 * e_phonon_eff_e * len(phonons)
        else:
            raise ValueError("Unexpected width model.")

        delta_zpe = zpe_gs - zpe_es
        result_store["delta_zpe"] = delta_zpe
        if abs(delta_zpe) > 1.:
            warnings.warn(f"Delta ZPE is {delta_zpe} eV, there is probably a big problem and the result may be garbage.")
        zpl -= delta_zpe

    logger.info(
        f"omega_gs = {e_phonon_eff * 1000} meV {e_phonon_eff * eV_in_J / cm1_in_J} cm-1"
    )

    result_store["omega_gs"] = e_phonon_eff

    logger.info(
        f"omega_es = {e_phonon_eff_e * 1000} meV {e_phonon_eff_e * eV_in_J / cm1_in_J} cm-1"
    )

    result_store["omega_es"] = e_phonon_eff_e

    logger.info(f"S_em = {S_em}")
    logger.info(f"d_fc^g,v = {dfcg_vib}")

    result_store["S_em"] = S_em

    result_store["d_fc^g,v"] = dfcg_vib

    if shape == LineShape.GAUSSIAN:
        fwhm = sig * sigma_to_fwhm
        fwhm0 = sig0 * sigma_to_fwhm
        logger.info(f"FWHM {fwhm * 1000} meV")
        logger.info(f"FWHM 0K {fwhm0 * 1000} meV")
    elif shape == LineShape.LORENTZIAN:
        fwhm = -sig * sigma_to_fwhm
        fwhm0 = -sig0 * sigma_to_fwhm
        logger.info(f"FWHM {-fwhm * 1000} meV")
        logger.info(f"FWHM 0K {-fwhm0 * 1000} meV")
    else:
        fwhm = None

    result_store["fwhm"] = fwhm
    result_store["fwhm0"] = fwhm0

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
    hrs = get_HR_factors(phonons, delta_R * 1e-10, bias=bias_si)
    S = np.sum(hrs)

    # array of mode specific pulsations/radial frequencies
    energies = get_energies(phonons, bias=bias_si)

    freqs = energies / h_si

    s_t = _get_s_t_raw(t, freqs, hrs)

    if pre_convolve is not None:
        sigma_freq = pre_convolve * eV_in_J / h_si / sigma_to_fwhm
        g = _gaussian(t, 1 / (two_pi * sigma_freq))
        if np.max(g) > 0:
            s_t *= g / np.max(g)

    exp_s_t = np.exp(s_t - S)

    g_t = exp_s_t * np.exp(1.0j * two_pi * t * zpl * eV_in_J / h_si)

    if sigma_si is None:
        logger.info("Using no line shape.")
        line_shape = np.ones(t.shape, dtype=complex)
    elif sigma_si <= 0:
        logger.info("Using a Lorentzian line shape.")
        sigma_freq = two_pi * sigma_si / h_si
        line_shape = np.array(np.exp(sigma_freq * np.abs(t)), dtype=complex)
    else:
        logger.info("Using a Gaussian line shape.")
        sigma_freq = two_pi * sigma_si / h_si
        line_shape = np.sqrt(2) * np.array(_gaussian(t, 1 / sigma_freq), dtype=complex)

    a_t = _window(g_t * line_shape, fn=window_fn)

    e = np.arange(0, N) * resolution_e
    A = fft.fft(a_t)

    I = e**3 * A

    return e, I


def get_HR_factors(phonons, delta_R_tot, bias=0):
    """
    delta_R_tot in SI
    """
    return np.array([ph.huang_rhys(delta_R_tot)
                     for ph in phonons
                     if ph.real
                     if ph.energy >= bias])


def get_energies(phonons, bias=0):
    """Return an array of mode energies in SI"""
    return np.array([ph.energy
                     for ph in phonons
                     if ph.real
                     if ph.energy >= bias])


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


def fc_spectra(phonons, delta_R, n_points=5000, disp=1):
    f, s, dirac_s = _stick_smooth_spectra(
        phonons, delta_R, lambda hr, e: hr * e, n_points, disp=disp
    )

    return f, f * s, f * dirac_s


def hr_spectra(phonons, delta_R, n_points=5000, disp=1):
    return _stick_smooth_spectra(
        phonons, delta_R, lambda hr, _e: hr, n_points, disp=disp
    )


def _stick_smooth_spectra(phonons, delta_R, height, n_points, disp=1):
    """
    delta_R in A
    """
    ph_e_meV = get_energies(phonons) * 1000 / eV_in_J

    mi = min(ph_e_meV)
    ma = max(ph_e_meV)

    e_meV = np.linspace(mi, ma + 0.2 * (ma - mi), n_points)

    w = 2 * (ma - mi) / n_points

    fc_spec = np.zeros(e_meV.shape)
    fc_sticks = np.zeros(e_meV.shape)

    hrs = get_HR_factors(phonons, delta_R * 1e-10)

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


def sigme_hybrid(T, S, e_phonon, e_phonon_e):
    return np.sqrt(np.sum([
        sigma_soft(T, S_i, e_i, e_phonon_e)**2
        for S_i, e_i in zip(S, e_phonon)
    ]))


def sigma_full_nd(T, delta_R, modes_gs, modes_es):
    """
    WIP
    There is still something missing in my reasoning
    sig2 = np.array((len(modes_gs),))

    mat_es = np.array([
        m.delta.reshape((-1,))
        for m in modes_es
    ])
    omegas_es = np.array([m.energy for m in modes_es])

    for i, m in enumerate(modes_es):
        # TODO Check units !!!!
        S_i = m.huang_rhys(delta_R * 1e-10)
        es_i = m.energy
        es_at_i = mat_es.dot(m.delta.reshape((-1,))).dot(omegas_es)
        sig2[i] = sigma_soft(T, S_i, e_i, es_at_i)**2

    return np.sqrt(np.sum(sig2))
    """
    raise NotImplementedError("This is not yet implemented")


def freq_from_finite_diff(left, mid, right, mu, A=0.01):
    """Compute a vibration energy from three energy points.

    :param left: energy (eV) of the left point
    :param mid: energy (eV) of the middle point
    :param right: energy (eV) of the right point
    :param mu: effective mass associated with the displacement from the middle
      point to the sides.
    :param A: (optional, 0.01) amplitude (A) of the displacement between the
      middle point and the sides.
    """
    curvature = (left + right - 2 * mid) * eV_in_J / (A * 1e-10)**2
    e_vib = hbar_si * np.sqrt(2 * curvature / mu)
    return e_vib / eV_in_J  # eV
