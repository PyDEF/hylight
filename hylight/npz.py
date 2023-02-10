"""Serialization of modes to numpy zipped file.

License
    Copyright (C) 2023  PyDEF development team

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import pickle
import os

import numpy as np

from .mode import Mode
from .constants import atomic_mass, eV_in_J


def archive_modes(modes, dest, compress=False):
    """Store modes in dest using numpy's npz format.

    :param modes: a list of Mode objects.
    :param dest: the path to write the modes to.
    :return: the data returned by load_phonons.
    """

    if isinstance(modes, tuple) and len(modes) == 3 and isinstance(modes[0], list):
        ph, _, _ = modes
    elif isinstance(modes, list) and isinstance(modes[0], Mode):
        ph = modes

    if not ph:
        raise ValueError("There are no modes to be stored.")

    n = len(ph)
    m = len(ph[0].ref)

    if not all(len(mode.ref) == m for mode in ph):
        raise ValueError("Some of the modes have differently shaped position.")

    if not all(len(mode.eigenvector) == m for mode in ph):
        raise ValueError("Some of the modes have differently shaped eigenvector.")

    if compress:
        save = np.savez_compressed
    else:
        save = np.savez

    eigenvectors = np.ndarray((n, m, 3))
    energies = np.ndarray((n,))

    for i, m in enumerate(ph):
        # store in meV
        energies[i] = (1 if m.real else -1) * m.energy * 1e3 / eV_in_J
        eigenvectors[i, :, :] = m.eigenvector

    with open(dest, mode="wb") as f:
        save(
            f,
            atoms=np.array([s.encode("ascii") for s in ph[0].atoms]),
            masses=ph[0].masses / atomic_mass,
            ref=ph[0].ref,
            eigenvectors=eigenvectors,
            energies=energies,
        )


def load_phonons(source):
    """Load modes from a Hylight archive."""

    _, ext = os.path.splitext(source)

    if ext == ".npz":
        return load_phonons_npz(source)
    else:
        return load_phonons_pickle(source)


def load_phonons_npz(source):
    with np.load(source) as f:
        atoms = [blob.decode("ascii") for blob in f["atoms"]]

        # The [:] is used to force the read and convert to a numpy array.
        masses = f["masses"]
        ref = f["ref"]
        vecs = f["eigenvectors"]
        enes = f["energies"]

        phonons = []

        for i, (v, e) in enumerate(zip(vecs, enes)):
            phonons.append(Mode(atoms, i, e >= 0, abs(e), ref, v[:], masses))

    return (phonons, *pops_and_masses(phonons))


def load_phonons_pickle(source, gz=False):
    """Load modes from a pickled file."""
    with open(source, "rb") as f:
        try:
            label, data = pickle.load(f)
        except ValueError:
            raise ValueError("This is not a pickled mode file.")

    # legacy format
    if label != "hylight-pkl-modes":
        raise ValueError("This is not a pickled mode file.")

    return data


def pops_and_masses(modes):
    if not modes:
        return [], []

    masses = modes[0].masses / atomic_mass
    pops = {sp: modes[0].atoms.count(sp) for sp in set(modes[0].atoms)}

    return pops, masses
