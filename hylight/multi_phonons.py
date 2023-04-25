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
    :param mask: a :class:`Mask` instance to select modes base on frequencies
    :param shape: the lineshape (a :class:`LineShape` instance)
    :param pre_convolve: (float, optional, None) if not None, standard deviation of a pre convolution gaussian
    :param window_fn: windowing function in the form provided by numpy (see :func:`numpy.hamming`)
    :returns: :code:`(energy_array, intensity_array)`
    """

    if e_max is None:
        e_max = zpl * 2.5

    if e_max < 2 * zpl:
        raise ValueError(
            f"e_max = {e_max} < 2 * zpl = {2 * zpl}: this will cause excessive numerical artifacts."
        )

    if isinstance(phonons, str):
        phonons, _, _ = load_phonons(phonons)

    if isinstance(delta_R, tuple):
        pos_gs, pos_es = delta_R
        delta_R = compute_delta_R(pos_gs, pos_es)

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

    if pre_convolve is not None:
        sigma_freq = pre_convolve * eV_in_J / h_si / sigma_to_fwhm
        g = gaussian(t, 1 / (two_pi * sigma_freq))
        if np.max(g) > 0:
            s_t *= g / np.max(g)

    exp_s_t = np.exp(s_t - S)

    g_t = exp_s_t * np.exp(1.0j * two_pi * t * zpl * eV_in_J / h_si)

    line_shape = make_line_shape(t, sigma_si, shape)

    a_t = _window(g_t * line_shape, fn=window_fn)

    e = np.arange(0, N) * resolution_e
    A = fft.fft(a_t)

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
        raise ValueError(f"Unimplemented or unkown lineshape {shape}.")


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

    Use :func:`hylight.multi_phonons.rect` for as a dummy window.
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
    """A dummy windowing function that works like numpy.hamming, but as no effect on data."""
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


class Mask:
    "An energy based mask for the set of modes."

    def __init__(self, intervals):
        self.intervals = intervals

    @classmethod
    def from_bias(cls, bias):
        """Create a mask that reject modes of energy between 0 and `bias`.

        :param bias: minimum of accepted energy (eV)
        :returns: a fresh instance of `Mask`.
        """
        if bias > 0:
            return cls([(0, bias * eV_in_J)])
        else:
            return cls([])

    def add_interval(self, interval):
        "Add a new interval to the mask."
        assert (
            isinstance(interval, tuple) and len(tuple) == 2
        ), "interval must be a tuple of two values."
        self.intervals.append(interval)

    def as_bool(self, ener):
        "Convert to a boolean `np.ndarray` based on `ener`."
        bmask = np.ones(ener.shape, dtype=bool)

        for bot, top in self.intervals:
            bmask &= (ener < bot) | (ener > top)

        return bmask

    def accept(self, value):
        "Return True if value is not under the mask."
        return not any(bot <= value <= top for bot, top in self.intervals)

    def reject(self, value):
        "Return True if `value` is under the mask."
        return any(bot <= value <= top for bot, top in self.intervals)

    def plot(self, ax, unit):
        """Add a graphical representation of the mask to a plot.

        :param ax: a matplotlib `Axes` object.
        :param unit: the unit of energy to use (ex: :attr:`hylight.constant.eV_in_J` if the plot uses eV)
        :returns: a function that must be called without arguments after resizing the plot.
        """
        from matplotlib.patches import Rectangle

        rects = []
        for bot, top in self.intervals:
            p = Rectangle((bot / unit, 0), (top - bot) / unit, 0, facecolor="grey")
            ax.add_patch(p)
            rects.append(p)

        def resize():
            (_, h) = ax.transAxes.transform((0, 1))
            for r in rects:
                r.set(height=h)

        return resize
