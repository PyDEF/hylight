"Semi classical guess of linewidth."
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
from enum import Enum
from math import floor
import numpy as np
from scipy import fft

from matplotlib import pyplot as plt

from .mode import get_energies, get_HR_factors, rot_c_to_v
from .loader import load_phonons
from .constants import atomic_mass, eV_in_J, hbar_si, h_si, kb_eV, sigma_to_fwhm
from .multi_phonons import (
    compute_delta_R,
    _get_s_t_raw,
    make_line_shape,
    _window,
    LineShape,
)

import logging

logger = logging.getLogger("hylight")


class OmegaEff(Enum):
    r"""Mode of computation of a effective frequency.

    :attr:`FC_MEAN`:

    .. math::
        \Omega = \frac{\sum_j \omega_j d^\text{FC}_j}{\sum_j d^\text{FC}_j}

    Should be used with :attr:`ExPES.ISO_SCALE` because it is associated with
    the idea that all the directions are softened equally in the excited state.

    :attr:`HR_MEAN`:

    .. math::
        \Omega = \frac{d^\text{FC}}{S_\text{tot}} = \frac{\sum_j \omega_j S_j}{\sum_j S_j}

    :attr:`HR_RMS`:

    .. math::
        \Omega = \sqrt{\frac{\sum_j \omega_j^2 S_j}{\sum_j S_j}}

    :attr:`FC_RMS`:

    .. math::
        \Omega = \sqrt{\frac{\sum_j \omega_j^2 d^\text{FC}_j}{\sum_j d^\text{FC}_j}}

    Should be used with :attr:`ExPES.SINGLE_ES_FREQ` because it makes sense
    when we get only one Omega_eff for the excited state (this single
    effective frequency should be computed beforehand)

    :attr:`ONED_FREQ`:

    .. math::
        \Omega = \frac{(\Delta Q^T D \Delta Q)}{{\Delta Q}^2}

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

    :attr:`ISO_SCALE`: We suppose eigenvectors are the same, but the frequencies are
    scaled by a constant factor.

    :attr:`SINGLE_ES_FREQ`: We suppose all modes have the same frequency (that should be provided).

    :attr:`FULL_ND`: (Not implemented) We know the frequency of each ES mode.
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


def expected_width(
    phonons,
    delta_R,
    fc_shift_gs,
    fc_shift_es,
    T,
    mask=None,
    omega_eff_type=OmegaEff.FC_MEAN,
    ex_pes=ExPES.ISO_SCALE,
):
    """Compute a spectrum without free parameters.

    :param phonons: list of modes
    :param delta_R: displacement in A
    :param fc_shift_gs: Ground state/absorption Franck-Condon shift in eV
    :param fc_shift_es: Exceited state/emmission Franck-Condon shift in eV
    :param T: temperature in K
    :param resolution_e: energy resolution in eV
    :param bias: ignore low energy vibrations under bias in eV
    :param window_fn: windowing function in the form provided by numpy (see numpy.hamming)
    :param pre_convolve: (float, optional, None) if not None, standard deviation of the pre convolution gaussian
    :param shape: ZPL line shape.
    :param omega_eff_type: mode of evaluation of effective frequency.
    :param result_store: a dictionary to store some intermediate results.
    :param ex_pes: mode of evaluation of the ES PES curvature.
    :returns: :code:`(sig(T=0), sig(T))`
    """

    hrs = get_HR_factors(phonons, delta_R * 1e-10, mask=mask)
    es = get_energies(phonons, mask=mask) / eV_in_J

    S_em = np.sum(hrs)
    logger.info(f"S = {S_em}")
    dfcg_vib = np.sum(hrs * es)

    e_phonon_eff = effective_phonon_energy(
        omega_eff_type, hrs, es, phonons[0].masses / atomic_mass, delta_R
    )

    if ex_pes.omega is None:
        if fc_shift_gs is None or fc_shift_gs < 0:
            raise ValueError(
                "fc_shift_gs must not be omited unless an effective frequency for the excited state is provided."
            )
        if fc_shift_es is None:
            raise ValueError(
                "fc_shift_es must not be omited unless an effective frequency for the excited state is provided."
            )
        alpha = np.sqrt(fc_shift_es / fc_shift_gs)
        e_phonon_eff_e = e_phonon_eff * alpha
        logger.info(f"d_fc^e,v = {dfcg_vib * alpha**2}")
    else:
        e_phonon_eff_e = ex_pes.omega
        alpha = e_phonon_eff_e / e_phonon_eff
        logger.info(f"d_fc^e,v = {dfcg_vib * alpha**2}")

    if ex_pes == ExPES.ISO_SCALE:
        sig = sigma(T, S_em, e_phonon_eff, e_phonon_eff_e)
        sig0 = sigma(0, S_em, e_phonon_eff, e_phonon_eff_e)
    elif ex_pes == ExPES.SINGLE_ES_FREQ:
        if ex_pes.omega is not None:
            e_phonon_eff_e = ex_pes.omega

        sig = sigma_hybrid(T, hrs, es, e_phonon_eff_e)
        sig0 = sigma_hybrid(0, hrs, es, e_phonon_eff_e)
    else:
        raise ValueError("Unexpected width model.")

    return sig0 * sigma_to_fwhm, sig * sigma_to_fwhm


def guess_width(
    phonons,
    delta_R,
    fc_shift_gs,
    fc_shift_es,
    T,
    resolution_e=1e-3,
    mask=None,
    shape=LineShape.GAUSSIAN,
    omega_eff_type=OmegaEff.FC_MEAN,
    ex_pes=ExPES.ISO_SCALE,
    window_fn=np.hamming,
):
    """Try to guess the width of the line from a 1D semi-classical model.

    :param phonons: list of modes (see :func:`load_phonons`) or path to load the modes
    :param delta_R: displacement in A in a numpy array (see :func:`compute_delta_R`)
      or tuple of two paths :code:`(pos_gs, pos_es)`
    :param fc_shift_gs: Ground state/absorption Franck-Condon shift in eV
    :param fc_shift_es: Exceited state/emmission Franck-Condon shift in eV
    :param T: temperature in K
    :param resolution_e: energy resolution in eV
    :param mask: a mask used in other computations to show on the plot.
    :param shape: the lineshape (a :class:`LineShape` instance)
    :param omega_eff_type: mode of evaluation of effective frequency.
    :param ex_pes: mode of evaluation of the ES PES curvature.
    :param window_fn: windowing function in the form provided by numpy (see numpy.hamming)
    :returns: a guess width in eV
    """

    if isinstance(phonons, str):
        phonons, _, _ = load_phonons(phonons)

    if isinstance(delta_R, tuple):
        pos_gs, pos_es = delta_R
        delta_R = compute_delta_R(pos_gs, pos_es)

    e_max = 3

    sample_rate = 2 * e_max * eV_in_J / h_si

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

    g_t = np.exp(s_t - S)

    trial_line_sig = 20e-3
    trial_line_shape = make_line_shape(t, trial_line_sig * eV_in_J, shape)

    a_t = _window(g_t * trial_line_shape * np.exp(1.0j * t * 3 * eV_in_J / hbar_si), fn=window_fn)

    e = np.arange(int(floor(-N / 2)), int(floor(N / 2))) * resolution_e
    trial_A = np.abs(fft.fft(a_t))
    trial_A /= integrate(e, trial_A)
    trial_sig2 = variance(e, trial_A)

    _, ex_fwhm = expected_width(
        phonons,
        delta_R,
        fc_shift_gs,
        fc_shift_es,
        T,
        mask=mask,
        omega_eff_type=omega_eff_type,
        ex_pes=ex_pes,
    )

    expected_sig2 = (ex_fwhm / sigma_to_fwhm) ** 2

    expected_line_sig2 = expected_sig2 + trial_line_sig**2 - trial_sig2

    if expected_line_sig2 <= 0.:
        raise ValueError("Guessed line variance is not positive.")

    return np.sqrt(expected_line_sig2) * sigma_to_fwhm


def integrate(x, y):
    r"""Integrate a function over x.

    :param x: :class:`numpy.ndarray` of x values
    :param y: :code:`y = f(x)`
    :returns: :math:`\int f(x) dx`
    """
    return np.trapz(y, x=x)


def variance(x, y):
    "Compute the variance of a random variable of distribution y."
    return integrate(x, x**2 * y) - integrate(x, x * y) ** 2


def sigma(T, S_em, e_phonon_g, e_phonon_e):
    """Temperature dependant standard deviation of the lineshape.

    :param T: temperature in K
    :param S_em: emmission Huang-Rhys factor
    :param e_phonon_g: energy of the GS PES vibration (eV)
    :param e_phonon_e: energy of the ES PES vibration (eV)
    """
    coth = 1.0 / np.tanh(e_phonon_e / (2 * kb_eV * T)) if T > 0.0 else 1.0

    return np.sqrt(S_em * e_phonon_g**3 / e_phonon_e * coth)


def sigma_hybrid(T, S, e_phonon, e_phonon_e):
    """Compute the width of the ZPL for the :attr:`ExPES.SINGLE_ES_FREQ` mode."""
    return np.sqrt(
        np.sum([sigma(T, S_i, e_i, e_phonon_e) ** 2 for S_i, e_i in zip(S, e_phonon)])
    )


def duschinsky(phonons_a, phonons_b):
    r"""Dushinsky matrix from b to a :math:`S_{a \\gets b}`."""
    return rot_c_to_v(phonons_a) @ rot_c_to_v(phonons_b).transpose()


def sigma_full_nd(T, delta_R, modes_gs, modes_es, mask=None):
    """Compute the width of the ZPL for the :attr:`ExPES.FULL_ND` mode.

    :param T: temperature in K.
    :param delta_R: distorsion in A.
    :param modes_gs: list of Modes of the ground state.
    :param modes_es: list of Modes of the excited state.
    :returns: :class:`numpy.ndarray` with only the width for the modes that are real in ground state.
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
            sigma(T, S_i, e_g_i, e_e_i)
            for S_i, m, e_g_i, e_e_i in zip(S, modes_gs, e_g, e_e)
            if m.real and mask.accept(m.energy)
        ]
    )


def effective_phonon_energy(omega_eff_type, hrs, es, masses, delta_R=None):
    """Compute an effective phonon energy in eV following the strategy of omega_eff_type.

    :param omega_eff_type: The mode of evaluation of the effective phonon energy.
    :param hrs: The array of Huang-Rhyes factor for each mode.
    :param es: The array of phonon energy in eV.
    :param masses: The array of atomic masses in atomic mass unit.
    :param delta_R: The displacement between GS and ES in A.
        It is only required if omega_eff_type is ONED_FREQ.
    :returns: The effective energy in eV.
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
