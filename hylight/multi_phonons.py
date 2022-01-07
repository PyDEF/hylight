"""
FIXME This does not work properly just now.
"""
import numpy as np
import scipy.fft

from .loader import load_phonons, load_poscar
from .constants import h_si, two_pi, eV_in_J, THz_in_meV

debug_thing = [None]


def spectra(
    outcar, poscar_gs, poscar_es, zpl, sigma=None, resolution_e=1e-5, N=1_000_000
):
    """
    outcar, poscar_es, poscar_gs are path
    zpl is in eV
    sigma is in s-1 ?
    """

    phonons, _, _ = load_phonons(outcar)
    delta_R = compute_delta_R(poscar_gs, poscar_es)

    return compute_spectra(
        phonons, delta_R, zpl, sigma=sigma, resolution_e=resolution_e, N=N
    )


def compute_spectra(phonons, delta_R_tot, zpl, sigma, resolution_e, N):
    """
    zpl in eV
    delta_R_tot in A
    sigma in s-1
    resolution_e in eV
    """

    resolution_f = resolution_e * eV_in_J / h_si
    resolution_t = 1.0 / (resolution_f * N)

    print(resolution_t)

    top = N * resolution_t

    t = np.arange(0, N) * resolution_t
    # t = np.linspace(-top, top, N)
    # t = np.arange(-N // 2, (N - N // 2)) * resolution_t

    hrs = get_HR_factors(phonons, delta_R_tot * 1e-10)
    S = np.sum(hrs)

    pulses = two_pi * get_energies(phonons) / h_si

    # Fourier transform of individual S_i \delta {(\nu - \nu_i)}
    s_i_t = hrs.reshape((1, -1)) * np.exp(
        -1.0j * pulses.reshape((1, -1)) * t.reshape((-1, 1))
    )
    s_t = np.sum(s_i_t, axis=1)
    exp_s_t = np.exp(s_t)

    if sigma is None:
        line_shape = np.ones(t.shape, dtype=complex)
    elif sigma < 0:
        line_shape = np.array(np.exp(sigma * np.abs(t)), dtype=complex)
    else:
        line_shape = np.array(gaussian(t, 4.0 / sigma), dtype=complex)

    g_t = exp_s_t * np.exp(1.0j * two_pi * zpl * eV_in_J * t / h_si) * np.exp(-S)

    a_t = window(g_t * line_shape, fn=nuttall)

    nu = scipy.fft.fftfreq(N, resolution_f)
    A = scipy.fft.fft(a_t)

    nu = scipy.fft.fftshift(nu) * resolution_e / resolution_f
    A = scipy.fft.fftshift(A)

    I = nu ** 3 * A

    debug_thing[0] = locals()

    return nu, I


def gaussian(e, sigma):
    return np.exp(-(e ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(two_pi))


def get_HR_factors(phonons, delta_R_tot):
    return np.array([ph.huang_rhys(delta_R_tot) for ph in phonons])


def get_energies(phonons):
    """
    Return energies in SI
    """
    return np.array([ph.energy for ph in phonons])


def compute_delta_R(poscar_gs, poscar_es):
    """Return $\\Delta R$ in A."""
    return load_poscar(poscar_gs) - load_poscar(poscar_es)


def nuttall(i, n):
    a0 = 0.355768
    a1 = 0.487396
    a2 = 0.144232
    a3 = 0.012604
    return (
        a0
        - a1 * np.cos(two_pi * i / n)
        + a2 * np.cos(2 * two_pi * i / n)
        - a3 * np.cos(3 * two_pi * i / n)
    )


def hann(i, n):
    return np.sin(i * two_pi * 0.5 / n) ** 2


def rect(i, n):
    return np.ones(i.shape)


def window(data, fn=nuttall):
    n = len(data)

    i = np.arange(0, n)
    return data * fn(i, n)
