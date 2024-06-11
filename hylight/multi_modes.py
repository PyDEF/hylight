"""Simulation of spectra in nD model.
"""

# Copyright (c) 2024, Théo Cavignac <theo.cavignac+dev@gmail.com>, The PyDEF team <camille.latouche@cnrs-imn.fr>
# Licensed under the EUPL
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
    kb,
)
from .utils import periodic_diff, gaussian

import logging

logger = logging.getLogger("hylight")


class LineShape(Enum):
    "Line shape type."
    GAUSSIAN = 0
    LORENTZIAN = 1
    NONE = 2


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
    :returns: :code:`(figure, (ax_FC, ax_S))`
    """
    from matplotlib import pyplot as plt

    phonons, _, _ = load_phonons(mode_source)
    delta_R = compute_delta_R(poscar_gs, poscar_es)

    updaters = []

    if not any(p.real for p in phonons):
        raise ValueError("No real mode extracted.")

    f, fc, dirac_fc = fc_spectrum(phonons, delta_R, disp=disp)
    f, s, dirac_s = hr_spectrum(phonons, delta_R, disp=disp)

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


def compute_spectrum(
    phonons,
    delta_R,
    zpl,
    fwhm,
    e_max=None,
    resolution_e=1e-3,
    mask=None,
    shape=LineShape.GAUSSIAN,
    pre_convolve=None,
    load_phonons=load_phonons,
    window_fn=np.hamming,
    T=0,
):
    """Compute a luminescence spectrum with the time-dependant formulation with an arbitrary linewidth.

    :param phonons: list of modes (see :func:`load_phonons`) or path to load the modes
    :param delta_R: displacement in A in a numpy array (see :func:`compute_delta_R`)
      or tuple of two paths :code:`(pos_gs, pos_es)`
    :param zpl: zero phonon line energy in eV
    :param fwhm: ZPL lineshape full width at half maximum in eV or None

        - if :code:`fwhm is None or fwhm == 0.0`: the raw spectrum is provided unconvoluted
        - if :code:`fwhm > 0`: the spectrum is convoluted with a gaussian line shape
        - if :code:`fwhm < 0`: error

    :param e_max: (optional, :code:`2.5*e_zpl`) max energy in eV (should be at greater than :code:`2*zpl`)
    :param resolution_e: (optional, :code:`1e-3`) energy resolution in eV
    :param load_phonons: a function to read phonons from files.
    :param mask: a :class:`.mode.Mask` instance to select modes base on frequencies
    :param shape: the lineshape (a :class:`LineShape` instance)
    :param pre_convolve: (float, optional, None) if not None, standard deviation of a pre convolution gaussian
    :param window_fn: windowing function in the form provided by numpy (see :func:`numpy.hamming`)
    :returns: :code:`(energy_array, intensity_array)`
    """

    if e_max is None:
        e_max = zpl * 3.0

    if e_max < 2 * zpl:
        raise ValueError(
            f"e_max = {e_max} < 2 * zpl = {2 * zpl}: this will cause excessive numerical artifacts."
        )

    if isinstance(phonons, str):
        phonons, _, _ = load_phonons(phonons)

    if isinstance(delta_R, tuple):
        pos_gs, pos_es = delta_R
        delta_R = compute_delta_R(pos_gs, pos_es)

        pos_gs, lattice_gs = load_poscar_latt(pos_gs)

        warn = None
        warn_bad = None

        for ph in phonons:
            d = np.linalg.norm(periodic_diff(lattice_gs, pos_gs, ph.ref), axis=1)

            if np.max(d) > 5e-2:
                md = np.max(d)
                i0 = np.argmax(d)
                warn_bad = (ph.n, i0, md)
            elif np.max(d) > 5e-3:
                md = np.max(d)
                i0 = np.argmax(d)
                warn = (ph.n, i0, md)

        if warn_bad is not None:
            n, i0, md = warn_bad
            logger.error(
                f"Mode {ph.n} has a reference position very far from GS position. (atom {i0+1} moved by {md} A)"
            )
        elif warn is not None:
            n, i0, md = warn
            logger.warning(
                f"Mode {n} has a reference position somewhat far from GS position. (atom {i0+1} moved by {md})"
            )
    else:
        logger.warning(
            "Make sure that delta_R and phonons are described in the same cell."
        )

    if fwhm is None or fwhm == 0.0:
        sigma_si = None
    elif fwhm < 0.0:
        raise ValueError("FWHM cannot be negative.")
    else:
        sigma_si = fwhm * eV_in_J / sigma_to_fwhm

    sample_rate = e_max * eV_in_J / h_si

    resolution_t = 1 / sample_rate

    N = int(e_max / resolution_e)

    t = np.arange((-N) // 2 + 1, (N) // 2 + 1) * resolution_t

    # array of mode specific HR factors
    hrs = get_HR_factors(phonons, delta_R * 1e-10, mask=mask)
    S = np.sum(hrs)
    logger.info(f"Total Huang-Rhys factor {S}.")

    # array of mode specific pulsations/radial frequencies
    energies = get_energies(phonons, mask=mask)

    freqs = energies / h_si

    s_t = _get_s_t_raw(t, freqs, hrs)

    if T > 0:
        s_t += _get_c_t_raw(T, t, freqs, hrs)

    if pre_convolve is not None:
        sigma_freq = pre_convolve * eV_in_J / h_si / sigma_to_fwhm
        g = gaussian(t, 1 / (two_pi * sigma_freq))
        if np.max(g) > 0:
            s_t *= g / np.max(g)

    exp_s_t = np.exp(s_t - S)

    g_t = exp_s_t * np.exp(1.0j * two_pi * t * zpl * eV_in_J / h_si)

    line_shape = make_line_shape(t, sigma_si, shape)

    a_t = _window(g_t * line_shape, fn=window_fn)

    A = fft.fft(a_t)

    e = np.arange(0, N) * resolution_e
    I = np.abs(e**3 * A)  # noqa: E741

    return e, I / np.max(I)


def make_line_shape(t, sigma_si, shape):
    """Create the lineshape function in time space.

    :param t: the time array (in s)
    :param sigma_si: the standard deviation of the line
    :param shape: the type of lineshape (an instance of :class:`LineShape`)
    :returns: a :class:`numpy.ndarray` of the same shape as :code:`t`
    """

    if sigma_si is None:
        logger.info("Using no line shape.")
        return np.ones(t.shape, dtype=complex)
    elif shape == LineShape.LORENTZIAN:
        logger.info("Using a Lorentzian line shape.")
        sigma_freq = two_pi * sigma_si / h_si
        return np.array(np.exp(-sigma_freq * np.abs(t)), dtype=complex)
    elif shape == LineShape.GAUSSIAN:
        logger.info("Using a Gaussian line shape.")
        sigma_freq = two_pi * sigma_si / h_si
        return np.sqrt(2) * np.array(gaussian(t, 1 / sigma_freq), dtype=complex)
    else:
        raise ValueError(f"Unimplemented or unknown lineshape {shape}.")


def compute_delta_R(poscar_gs, poscar_es):
    """Return :math:`\\Delta R` in A.

    :param poscar_gs: path to ground state positions file.
    :param poscar_es: path to excited state positions file.
    :returns: a :class:`numpy.ndarray` of shape :code:`(n, 3)` where :code:`n` is the number of atoms.
    """

    pos1, lattice1 = load_poscar_latt(poscar_gs)
    pos2, lattice2 = load_poscar_latt(poscar_es)

    if np.linalg.norm(lattice1 - lattice2) > 1e-5:
        raise ValueError("Lattice parameters are not matching.")

    return periodic_diff(lattice1, pos1, pos2)


def _get_s_t_raw(t, freqs, hrs):
    # Fourier transform of individual S_i \delta {(\nu - \nu_i)}

    def slow():
        # Slower less memory intensive solution
        s_t = np.zeros(t.shape, dtype=complex)
        for hr, fr in zip(hrs, freqs):
            s_t += hr * np.exp(-1.0j * two_pi * fr * t)
        return s_t

    def fast():
        # This can create a huge array if freqs is too big
        # but it let numpy handle everything so it is really fast
        return hrs.reshape((1, -1)) * np.exp(
            -1.0j * two_pi * freqs.reshape((1, -1)) * t.reshape((-1, 1))
        )

    if len(freqs) * len(t) > 100e6:
        # above 100 million coefficients, don't even try
        return slow()
    else:
        try:
            s_i_t = fast()
        except MemoryError:
            return slow()
        else:
            # sum over the modes:
            return np.sum(s_i_t, axis=1)


def _get_c_t_raw(T, t, freqs, hrs):
    # Bose-Einstein statistics
    occs = 1.0 / (np.exp(freqs * h_si / (T * kb)) - 1.0)

    def slow():
        # Slower less memory intensive solution
        c_t = np.zeros(t.shape, dtype=float)
        for occ, hr, fr in zip(occs, hrs, freqs):
            c_t += 2.0 * occ * hr * (np.cos(two_pi * fr * t) - 1)
        return c_t

    def fast():
        # This can create a huge array if freqs is too big
        # but it let numpy handle everything so it is really fast
        return (
            2.0
            * (occs * hrs).reshape((1, -1))
            * (np.cos(two_pi * freqs.reshape((1, -1)) * t.reshape((-1, 1))) - 1)
        )

    if len(freqs) * len(t) > 100e6:
        # above 100 million coefficients, don't even try
        return slow()
    else:
        try:
            c_i_t = fast()
        except MemoryError:
            return slow()
        else:
            # sum over the modes:
            return np.sum(c_i_t, axis=1)


def _window(data, fn=np.hamming):
    """Apply a windowing function to the data.

    Use :func:`hylight.multi_modes.rect` for as a dummy window.
    """
    n = len(data)
    return data * fn(n)


def fc_spectrum(phonons, delta_R, n_points=5000, disp=1):
    "Build arrays for plotting a spectrum energy spectral function."
    f, fc, dirac_fc = _stick_smooth_spectrum(
        phonons, delta_R, lambda hr, e: hr * e, n_points, disp=disp
    )

    return f, fc, dirac_fc


def hr_spectrum(phonons, delta_R, n_points=5000, disp=1):
    "Build arrays for plotting a spectrum phonon spectral function."
    return _stick_smooth_spectrum(
        phonons, delta_R, lambda hr, _e: hr, n_points, disp=disp
    )


def _stick_smooth_spectrum(phonons, delta_R, height, n_points, disp=1):
    """Plot a spectra of Dirac's peaks with a smoothed background.

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
    "A dummy windowing function that works like numpy.hamming, but as no effect on data."
    return np.ones((n,))


def duschinsky(phonons_a, phonons_b):
    r"""Dushinsky matrix from b to a :math:`S_{a \\gets b}`."""
    return rot_c_to_v(phonons_a) @ rot_c_to_v(phonons_b).transpose()


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


def dynmatshow(dynmat, blocks=None):
    """Plot the dynamical matrix.

    :param dynmat: numpy array representing the dynamical matrice in SI.
    :param blocks: (optional, None) a list of coloured blocks in the form
        :code:`(label, number_of_atoms, color)`.
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
