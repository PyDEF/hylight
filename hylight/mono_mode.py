"""Simulation of luminescence spectra in 1D model.
"""
# Copyright (c) 2024, Théo Cavignac <theo.cavignac+dev@gmail.com>, The PyDEF team <camille.latouche@cnrs-imn.fr>
# Licensed under the EUPL
import logging

import numpy as np

from .utils import gaussian


logger = logging.getLogger("hylight")


def compute_spectrum(
    e_zpl,
    S,
    sig,
    e_phonon_g,
    e=None,
):
    """Compute a spectrum from 1D model with experimental like inputs.

    :param e_zpl: energy of the zero phonon line, in eV
    :param S: the emission Huang-Rhys factor
    :param sig: the lineshape standard deviation
    :e_phonon_g: the ground state phonon energy
    :param e: (optional, None) a numpy array of energies to compute the spectrum at
      if ommited, an array ranging from 0 to 3*e_zpl will be created.
    """

    if e is None:
        e = np.arange(0, 3 * e_zpl, 1e-3)

    sp = np.zeros(e.shape)

    khi_e_khi_g_squared = np.exp(-S)

    details = []

    n, r = 0, 1

    while r > 1e-8 and n < 100:
        c = e**3 * khi_e_khi_g_squared * gaussian(e_zpl - n * e_phonon_g - e, sig)
        details.append(c)
        sp += c
        r = np.max(c) / np.max(sp)
        n += 1
        khi_e_khi_g_squared *= S / n

    logger.info(f"Summed up to {n - 1} replicas.")

    return (
        e,
        sp / np.max(sp),
        [[c / np.max(sp) for c in details]],
    )


def huang_rhys(stokes_shift, e_phonon):
    """Huang-Rhys factor from the Stokes shift and the phonon energy."""
    return 0.5 * stokes_shift / e_phonon
