"Simulation of spectra in nD model."
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
import warnings
from enum import Enum
import numpy as np
from scipy import fft

from .mode import get_energies, get_HR_factors, rot_c_to_v
from .loader import load_phonons
from .vasp.loader import load_poscar_latt
from .constants import (
    two_pi,
    eV_in_J,
    h_si,
    cm1_in_J,
    sigma_to_fwhm,
    hbar_si,
    atomic_mass,
)
from .mono_phonon import sigma_soft
from .utils import periodic_diff, gaussian

import logging

logger = logging.getLogger("hylight")


class LineShape(Enum):
    "Line shape type."
    GAUSSIAN = 0
    LORENTZIAN = 1
    NONE = 2


class OmegaEff(Enum):
    r"""Mode of computation of a effective frequency.

    :py:attr:`FC_MEAN`: $\\Omega = <\\omega_i>_{d_{FC,i}}$
        should be used with :py:attr:`ExPES.ISO_SCALE` because it is associated with
        the idea that all the directions are softened equally in the excited state.

    :py:attr:`HR_MEAN`: $\\Omega = d_{FC,tot} / S_{tot} = <\\omega_i>_{S_i}$

    :py:attr:`HR_RMS`: $\\Omega = \\sqrt{<\\omega_i^2>_{S_i}}$

    :py:attr:`FC_RMS`: $\\Omega = \\sqrt{<\\omega_i^2>_{d_{FC,i}}}$
        should be used with :py:attr:`ExPES.SINGLE_ES_FREQ` because it makes sense
        when we get only one Omega_eff for the excited state (this single
        effective frequency should be computed beforehand)

    :py:attr:`ONED_FREQ`: $\\Omega = (\\Delta Q^T D \\Delta Q) / {\\Delta Q}^2$
        Correspond to the actual curvature in the direction of the displacement.
    """

    FC_MEAN = 0
    HR_MEAN = 1
    HR_RMS = 2
    FC_RMS = 3
    ONED_FREQ = 4


class _ExPES(Enum):
    ISO_SCALE = 0
    SINGLE_ES_FREQ = 1
    FULL_ND = 2


class ExPES:
    """Mode of approximation of the ES PES curvature.

    :py:attr:`ISO_SCALE`: We suppose eigenvectors are the same, but the frequencies are
      scaled by a constant factor.

    :py:attr:`SINGLE_ES_FREQ`: We suppose all modes have the same frequency (that should be provided).

    :py:attr:`FULL_ND`: (Not implemented) We know the frequency of each ES mode.
    """

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


def spectrum(
    mode_source,
    poscar_gs,
    poscar_es,
    zpl,
    T,
    fc_shift_gs=None,
    fc_shift_es=None,
    e_max=None,
    resolution_e=1e-4,
    bias=0,
    mask=None,
    load_phonons=load_phonons,
    pre_convolve=None,
    shape=LineShape.GAUSSIAN,
    omega_eff_type=OmegaEff.FC_MEAN,
    result_store=None,
    ex_pes=ExPES.ISO_SCALE,
    correct_zpe=False,
):
    """Compute a complete spectrum without free parameters.

    :param mode_source: path to the vibration computation output file (by default a pickled file)
    :param path_struct_gs: path to the ground state relaxed structure file (by default a POSCAR)
    :param path_struct_es: path to the excited state relaxed structure file (by default a POSCAR)
    :param zpl: zero phonon line energy in eV
    :param T: temperature in K
    :param fc_shift_gs: Ground state/absorption Franck-Condon shift in eV
    :param fc_shift_es: Exited state/emmission Franck-Condon shift in eV
    :param e_max: max energy in eV (should be at least > 2*zpl)
    :param resolution_e: energy resolution in eV
    :param bias: ignore low energy vibrations under bias in eV
    :param pre_convolve: (float, optional, None) if not None, standard deviation of the pre convolution gaussian
    :param load_phonons: a function that takes mode_source and produce a list of phonons. By default expect an mode_source.
    :returns: (energy_array, intensity_array)
    """

    if e_max is None:
        e_max = zpl * 2.5
    phonons, _, _ = load_phonons(mode_source)
    delta_R = compute_delta_R(poscar_gs, poscar_es)

    if fc_shift_es is None:
        fc_shift_es = fc_shift_gs

    if mask is None:
        mask = Mask.from_bias(bias)

    e, I = compute_spectrum_soft(  # noqa: E741
        phonons,
        delta_R,
        zpl,
        T,
        fc_shift_gs,
        fc_shift_es,
        e_max,
        resolution_e,
        mask=mask,
        pre_convolve=pre_convolve,
        shape=shape,
        omega_eff_type=omega_eff_type,
        result_store=result_store,
        ex_pes=ex_pes,
        correct_zpe=correct_zpe,
    )

    return e, np.abs(I)


def plot_spectral_function(
    mode_source,
    poscar_gs,
    poscar_es,
    load_phonons=load_phonons,
    use_cm1=False,
    disp=1,
    mpl_params=None,
    mask=None,
):
    """Plot a two panel representation of the spectral function of the distorsion.

    :param mode_source: path to the mode file (by default a pickle file)
    :param poscar_gs: path to the file containing the ground state atomic positions.
    :param poscar_es: path to the file containing the excited state atomic positions.
    :param load_phonons: a function to read mode_source.
    :param use_cm1: use cm1 as the unit for frequency instead of meV.
    :param disp: standard deviation of the gaussians in background in meV.
    :param mpl_params: dictionary of kw parameters for pyplot.plot.
    :param mask: a mask used in other computations to show on the plot.
    :returns: (figure, (ax_FC, ax_S))
    """
    from matplotlib import pyplot as plt

    phonons, _, _ = load_phonons(mode_source)
    delta_R = compute_delta_R(poscar_gs, poscar_es)

    updaters = []

    if not any(p.real for p in phonons):
        raise ValueError("No real mode extracted.")

    f, fc, dirac_fc = fc_spectrum(phonons, delta_R, disp=disp)
    f, s, dirac_s = hr_spectrum(phonons, delta_R, disp=disp)
    fc = s * f
    dirac_fc = dirac_s * f

    if use_cm1:
        f *= 1e-3 * eV_in_J / cm1_in_J
        unit = cm1_in_J
    else:
        unit = 1e-3 * eV_in_J

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

    if mask:
        updaters.append(mask.plot(ax_s, unit))

    ax_s.stackplot(f, s, **s_stack)
    ax_s.plot(f, dirac_s, **s_peaks)

    if mask:
        updaters.append(mask.plot(ax_fc, unit))

    ax_fc.stackplot(f, fc, **fc_stack)
    ax_fc.plot(f, dirac_fc, **fc_peaks)

    ax_s.set_ylabel("$S(\\hbar\\omega)$ (A. U.)")
    ax_fc.set_ylabel("FC shift (meV)")

    plt.subplots_adjust(hspace=0)

    if use_cm1:
        ax_bottom.set_xlabel("Wavenumber (cm$^{-1}$)")
    else:
        ax_bottom.set_xlabel("E (meV)")

    fig.set_size_inches((10, 9))

    def update():
        for fn in updaters:
            fn()

    update()

    return fig, (ax_fc, ax_s)


def compute_spectrum_soft(
    phonons,
    delta_R,
    zpl,
    T,
    fc_shift_gs,
    fc_shift_es,
    e_max,
    resolution_e,
    mask=None,
    window_fn=np.hamming,
    pre_convolve=None,
    shape=LineShape.GAUSSIAN,
    omega_eff_type=OmegaEff.FC_MEAN,
    result_store=None,
    ex_pes=ExPES.ISO_SCALE,
    correct_zpe=False,
):
    """Compute a spectrum without free parameters.

    :param phonons: list of modes
    :param delta_R: displacement in A
    :param zpl: zero phonon line energy in eV
    :param T: temperature in K
    :param fc_shift_gs: Ground state/absorption Franck-Condon shift in eV
    :param fc_shift_es: Exceited state/emmission Franck-Condon shift in eV
    :param e_max: max energy in eV (should be at least > 2*zpl)
    :param resolution_e: energy resolution in eV
    :param bias: ignore low energy vibrations under bias in eV
    :param window_fn: windowing function in the form provided by numpy (see numpy.hamming)
    :param pre_convolve: (float, optional, None) if not None, standard deviation of the pre convolution gaussian
    :param shape: ZPL line shape.
    :param omega_eff_type: mode of evaluation of effective frequency.
    :param result_store: a dictionary to store some intermediate results.
    :param ex_pes: mode of evaluation of the ES PES curvature.
    :param correct_zpe: correct the ZPL to take the zero point energy into account.
    :returns: (energy_array, intensity_array)
    """

    if result_store is None:
        result_store = {}

    hrs = get_HR_factors(phonons, delta_R * 1e-10, mask=mask)
    es = get_energies(phonons, mask=mask) / eV_in_J

    S_em = np.sum(hrs)
    dfcg_vib = np.sum(hrs * es)

    e_phonon_eff = effective_phonon_energy(
        omega_eff_type, hrs, es, phonons[0].masses / atomic_mass, delta_R
    )

    if ex_pes.omega is None:
        if fc_shift_gs is None or fc_shift_gs < 0:
            raise ValueError(
                "fc_shift_gs must not be omited unless an effective frequency for the excited state is provided for."
            )
        if fc_shift_es is None:
            raise ValueError(
                "fc_shift_es must not be omited unless an effective frequency for the excited state is provided for."
            )
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

        sig = sigma_hybrid(T, hrs, es, e_phonon_eff_e)
        sig0 = sigma_hybrid(0, hrs, es, e_phonon_eff_e)
    else:
        raise ValueError("Unexpected width model.")

    if correct_zpe:
        zpe_gs = 0.5 * np.sum(es)
        if ex_pes == ExPES.ISO_SCALE:
            gamma = e_phonon_eff_e / e_phonon_eff
            zpe_es = zpe_gs * gamma
        elif ex_pes == ExPES.SINGLE_ES_FREQ:
            zpe_es = 0.5 * e_phonon_eff_e * len(es)
        else:
            raise ValueError("Unexpected width model.")

        delta_zpe = zpe_gs - zpe_es
        result_store["delta_zpe"] = delta_zpe
        if abs(delta_zpe) > 1.0:
            warnings.warn(
                f"Delta ZPE is {delta_zpe} eV, there is probably a big problem and the result may be garbage."
            )
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

    return compute_spectrum(
        phonons,
        delta_R,
        zpl,
        fwhm,
        e_max,
        resolution_e,
        mask=mask,
        window_fn=window_fn,
        pre_convolve=pre_convolve,
        shape=shape,
    )


def compute_spectrum_width_ah(
    phonons_gs,
    phonons_es,
    delta_R,
    zpl,
    T,
    e_max,
    resolution_e,
    bias=0,
    mask=None,
    window_fn=np.hamming,
    pre_convolve=None,
    shape=LineShape.GAUSSIAN,
    result_store=None,
    correct_zpe=False,
):
    """Luminescence spectrum with a line width that takes the ES PES curvature into account.

    Uses the full Dushinsky matrix for the computation of the line width, but
    assume the Dushinsky matrix to be the identity in the computation of the FC
    integrals (thus ignoring mode mixing).

    :param phonons: list of modes
    :param delta_R: displacement in A
    :param zpl: zero phonon line energy in eV
    :param T: temperature in K
    :param fc_shift_gs: Ground state/absorption Franck-Condon shift in eV
    :param fc_shift_es: Exceited state/emmission Franck-Condon shift in eV
    :param e_max: max energy in eV (should be at least > 2*zpl)
    :param resolution_e: energy resolution in eV
    :param bias: ignore low energy vibrations under bias in eV
    :param window_fn: windowing function in the form provided by numpy (see numpy.hamming)
    :param pre_convolve: (float, optional, None) if not None, standard deviation of the pre convolution gaussian
    :param shape: ZPL line shape.
    :param omega_eff_type: mode of evaluation of effective frequency.
    :param result_store: a dictionary to store some intermediate results.
    :param ex_pes: mode of evaluation of the ES PES curvature.
    :param correct_zpe: correct the ZPL to take the zero point energy into account.
    :returns: (energy_array, intensity_array)
    """

    if result_store is None:
        result_store = {}

    if mask is None:
        mask = Mask.from_bias(bias)

    hrs = get_HR_factors(phonons_gs, delta_R * 1e-10, mask=mask)
    es = get_energies(phonons_gs, mask=mask) / eV_in_J

    S_em = np.sum(hrs)
    dfcg_vib = np.sum(hrs * es)

    sig = np.sqrt(
        np.sum(sigma_full_nd(T, delta_R, phonons_gs, phonons_es, mask=mask) ** 2)
    )
    sig0 = np.sqrt(
        np.sum(sigma_full_nd(0, delta_R, phonons_gs, phonons_es, mask=mask) ** 2)
    )

    if correct_zpe:
        es_es = get_energies(phonons_gs, mask=mask) / eV_in_J
        zpe_gs = 0.5 * np.sum(es)
        zpe_es = 0.5 * np.sum(es_es)

        delta_zpe = zpe_gs - zpe_es
        result_store["delta_zpe"] = delta_zpe
        if abs(delta_zpe) > 1.0:
            warnings.warn(
                f"Delta ZPE is {delta_zpe} eV, there is probably a big problem and the result may be garbage."
            )
        zpl -= delta_zpe

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

    return compute_spectrum(
        phonons_gs,
        delta_R,
        zpl,
        fwhm,
        e_max,
        resolution_e,
        mask=mask,
        window_fn=window_fn,
        pre_convolve=pre_convolve,
        shape=shape,
    )


def compute_spectrum(
    phonons,
    delta_R,
    zpl,
    fwhm,
    e_max,
    resolution_e,
    bias=0,
    mask=None,
    window_fn=np.hamming,
    pre_convolve=None,
    shape=LineShape.GAUSSIAN,
):
    """Compute a luminescence spectrum with the time-dependant formulation with an arbitrary linewidth.

    :param phonons: list of modes
    :param delta_R: displacement in A
    :param zpl: zero phonon line energy in eV
    :param fwhm: zpl lineshape full width at half maximum in eV or None
      if fwhm is None: the raw spectrum is not convoluted with a line shape
      if fwhm < 0: the spectrum is convoluted with a lorentizan line shape
      if fwhm > 0: the spectrum is convoluted with a gaussian line shape
    :param e_max: max energy in eV (should be at least > 2*zpl)
    :param resolution_e: energy resolution in eV
    :param bias: ignore low energy vibrations under bias in eV
    :param window_fn: windowing function in the form provided by numpy (see numpy.hamming)
    :param pre_convolve: (float, optional, None) if not None, standard deviation of the pre convolution gaussian
    :returns: (energy_array, intensity_array)
    """

    if fwhm is None:
        sigma_si = None
    else:
        sigma_si = fwhm * eV_in_J / sigma_to_fwhm

    sample_rate = e_max * eV_in_J / h_si

    resolution_t = 1 / sample_rate

    N = int(e_max / resolution_e)

    t = np.arange((-N) // 2 + 1, (N) // 2 + 1) * resolution_t

    # array of mode specific HR factors
    hrs = get_HR_factors(phonons, delta_R * 1e-10, mask=mask)
    S = np.sum(hrs)

    # array of mode specific pulsations/radial frequencies
    energies = get_energies(phonons, mask=mask)

    freqs = energies / h_si

    s_t = _get_s_t_raw(t, freqs, hrs)

    if pre_convolve is not None:
        sigma_freq = pre_convolve * eV_in_J / h_si / sigma_to_fwhm
        g = gaussian(t, 1 / (two_pi * sigma_freq))
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
        line_shape = np.sqrt(2) * np.array(gaussian(t, 1 / sigma_freq), dtype=complex)

    a_t = _window(g_t * line_shape, fn=window_fn)

    e = np.arange(0, N) * resolution_e
    A = fft.fft(a_t)

    I = e**3 * A  # noqa: E741

    return e, I


def compute_delta_R(poscar_gs, poscar_es):
    """Return $\\Delta R$ in A.

    :param poscar_gs: path to ground state positions file.
    :param poscar_es: path to excited state positions file.
    :return: a np.array of (n, 3) shape where n is the number of atoms.
    """

    pos1, lattice1 = load_poscar_latt(poscar_gs)
    pos2, lattice2 = load_poscar_latt(poscar_es)

    if np.linalg.norm(lattice1 - lattice2) > 1e-5:
        raise ValueError("Lattice parameters are not matching.")

    return periodic_diff(lattice1, pos1, pos2)


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


def _window(data, fn=np.hamming):
    """Apply a windowing function to the data.

    Use hylight.multi_phonons.rect for as a dummy window.
    """
    n = len(data)
    return data * fn(n)


def fc_spectrum(phonons, delta_R, n_points=5000, disp=1):
    f, s, dirac_s = _stick_smooth_spectrum(
        phonons, delta_R, lambda hr, e: hr * e, n_points, disp=disp
    )

    return f, f * s, f * dirac_s


def hr_spectrum(phonons, delta_R, n_points=5000, disp=1):
    return _stick_smooth_spectrum(
        phonons, delta_R, lambda hr, _e: hr, n_points, disp=disp
    )


def _stick_smooth_spectrum(phonons, delta_R, height, n_points, disp=1):
    """

    :param phonons: list of phonons
    :param delta_R: displacement in A
    :param height: height of sticks
    :param n_points: number of points
    :param disp: width of the gaussians
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
        g_thin = gaussian(e_meV - e, w)
        g_fat = gaussian(e_meV - e, disp)
        fc_sticks += h * g_thin / np.max(g_thin)  # g_thin should be h high
        fc_spec += h * g_fat  # g_fat has a g area

    return e_meV, fc_spec, fc_sticks


def rect(n):
    """A dummy windowing function that works like numpy.hamming, but as no effect on data."""
    return np.ones((n,))


def sigma_hybrid(T, S, e_phonon, e_phonon_e):
    """Compute the width of the ZPL for the ExPES.SINGLE_ES_FREQ mode."""
    return np.sqrt(
        np.sum(
            [sigma_soft(T, S_i, e_i, e_phonon_e) ** 2 for S_i, e_i in zip(S, e_phonon)]
        )
    )


def duschinsky(phonons_a, phonons_b):
    r"""Dushinsky matrix from b to a $S_{a \\gets b}$."""
    return rot_c_to_v(phonons_a) @ rot_c_to_v(phonons_b).transpose()


def sigma_full_nd(T, delta_R, modes_gs, modes_es, mask=None):
    """Compute the width of the ZPL for the ExPES.FULL_ND mode.

    :param T: temperature in K.
    :param delta_R: distorsion in A.
    :param modes_gs: list of Modes of the ground state.
    :param modes_es: list of Modes of the excited state.
    :returns: np.array with only the width for the modes that are real in ground state.
    """
    Dush_gs_to_es = duschinsky(modes_es, modes_gs)

    D_es = np.diag(
        [(1 if m.real else -1) * (m.energy / hbar_si) ** 2 for m in modes_es]
    )

    # Extract the \gamma_i^T @ D_es @ \gamma_i
    f2 = np.diagonal(Dush_gs_to_es.transpose() @ D_es @ Dush_gs_to_es)
    e_e = np.sqrt(np.where(f2 >= 0, f2, 0)) * hbar_si / eV_in_J
    e_e_im = np.sqrt(np.where(f2 < 0, -f2, 0)) * hbar_si / eV_in_J

    if not np.all(e_e_im * np.array([m.real for m in modes_gs]) < 1e-8):
        raise ValueError(
            "Some of the ground state eigenvectors correspond to negative curvature in excited state."
        )

    e_g = np.array([mode.energy for mode in modes_gs]) / eV_in_J
    S = get_HR_factors(modes_gs, delta_R * 1e-10, mask=mask)

    return np.array(
        [
            sigma_soft(T, S_i, e_g_i, e_e_i)
            for S_i, m, e_g_i, e_e_i in zip(S, modes_gs, e_g, e_e)
            if m.real and mask.accept(m.energy)
        ]
    )


def freq_from_finite_diff(left, mid, right, mu, A=0.01):
    """Compute a vibration energy from three energy points.

    :param left: energy (eV) of the left point
    :param mid: energy (eV) of the middle point
    :param right: energy (eV) of the right point
    :param mu: effective mass associated with the displacement from the middle
        point to the sides.
    :param A: amplitude (A) of the displacement between the
        middle point and the sides.
    """
    curvature = (left + right - 2 * mid) * eV_in_J / (A * 1e-10) ** 2
    e_vib = hbar_si * np.sqrt(2 * curvature / mu)
    return e_vib / eV_in_J  # eV


def effective_phonon_energy(omega_eff_type, hrs, es, masses, delta_R=None):
    """Compute an effective phonon energy in eV following the strategy of omega_eff_type.

    :param omega_eff_type: The mode of evaluation of the effective phonon energy.
    :param hrs: The array of Huang-Rhyes factor for each mode.
    :param es: The array of phonon energy in eV.
    :param masses: The array of atomic masses in atomic mass unit.
    :param delta_R: The displacement between GS and ES in A.
        It is only required if omega_eff_type is ONED_FREQ.
    :return: The effective energy in eV.
    """

    _es = es * eV_in_J
    fcs = hrs * _es

    S_em = np.sum(hrs)
    dfcg_vib = np.sum(fcs)

    if omega_eff_type == OmegaEff.FC_MEAN:
        return np.sum(fcs * _es) / dfcg_vib / eV_in_J
    elif omega_eff_type == OmegaEff.HR_MEAN:
        return dfcg_vib / S_em / eV_in_J
    elif omega_eff_type == OmegaEff.HR_RMS:
        return np.sqrt(np.sum(fcs * _es) / S_em) / eV_in_J
    elif omega_eff_type == OmegaEff.FC_RMS:
        return np.sqrt(np.sum(fcs * _es * _es) / dfcg_vib) / eV_in_J
    elif omega_eff_type == OmegaEff.ONED_FREQ:
        if delta_R is None:
            raise ValueError("delta_R is required when using the ONED_FREQ model.")

        delta_Q = (
            (masses * atomic_mass).reshape((-1, 1)) ** 0.5 * delta_R * 1e-10
        ).reshape((-1,))
        delta_Q_2 = delta_Q.dot(delta_Q)
        return (hbar_si * np.sqrt(np.sum(fcs) / delta_Q_2)) / eV_in_J

    raise ValueError("Unknown effective frequency strategy.")


def dynmatshow(dynmat, blocks=None):
    """Plot the dynamical matrix.

    :param dynmat: numpy array representing the dynamical matrice in SI.
    :param blocks: (optional) a list of coloured blocks in the form `(label,
        number_of_atoms, color)`.
    """
    from matplotlib.patches import Patch
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.pyplot as plt

    if blocks:
        atmat = np.zeros(dynmat.shape)

        colors = ["white"]
        legends = []

        off = 0
        for i, (at, n, col) in enumerate(blocks):

            atmat[off : off + 3 * n, off : off + 3 * n] = i + 1

            off += 3 * n
            colors.append(col)
            legends.append(Patch(facecolor=col, label=at))

        atcmap = LinearSegmentedColormap.from_list("atcmap", colors, 256)
        blacks = LinearSegmentedColormap.from_list("blacks", ["none", "black"], 256)
    else:
        blacks = "Greys"

    if blocks:
        plt.imshow(atmat, vmin=0, vmax=len(blocks), cmap=atcmap)
        plt.legend(handles=legends)

    y = np.abs(dynmat) * atomic_mass / eV_in_J * 1e-20

    im = plt.imshow(y, cmap=blacks)
    ax = plt.gca()

    ax.set_xlabel("(atoms $\\times$ axes) index")
    ax.set_ylabel("(atoms $\\times$ axes) index")

    fig = plt.gcf()
    cb = fig.colorbar(im)

    cb.ax.set_ylabel("Dynamical matrix (eV . A$^{-2}$ . m$_p^{-1}$)")

    return fig, im


class Mask:
    def __init__(self, intervals):
        self.intervals = intervals

    @classmethod
    def from_bias(cls, bias):
        if bias > 0:
            return cls([(0, bias * eV_in_J)])
        else:
            return cls([])

    def add_interval(self, interval):
        assert isinstance(interval, tuple) and len(tuple) == 2, "interval must be a tuple of two values."
        self.intervals.append(interval)

    def as_bool(self, ener):
        bmask = np.ones(ener.shape, dtype=bool)

        for bot, top in self.intervals:
            bmask &= (ener < bot) | (ener > top)

        return bmask

    def accept(self, value):
        return not any(bot <= value <= top for bot, top in self.intervals)

    def reject(self, value):
        return any(bot <= value <= top for bot, top in self.intervals)

    def plot(self, ax, unit):
        from matplotlib.patches import Rectangle

        rects = []
        for bot, top in self.intervals:
            p = Rectangle((bot / unit, 0), (top - bot) / unit, 0, facecolor="grey")
            ax.add_patch(p)
            rects.append(p)

        def resize():
            (_, h) = ax.transAxes.transform([(0, 1)])
            for r in rects:
                r.set(height=h)

        return resize
