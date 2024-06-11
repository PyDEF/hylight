"""Module to load phonons frequencies and eigenvectors from phonopy output files.

It always uses PyYAML, but it may also need h5py to read hdf5 files.
"""

# Copyright (c) 2024, Théo Cavignac <theo.cavignac+dev@gmail.com>, The PyDEF team <camille.latouche@cnrs-imn.fr>
# Licensed under the EUPL
from __future__ import annotations

from os.path import isfile, join
from itertools import groupby
import gzip
from dataclasses import dataclass

import numpy as np

from ..constants import THz_in_meV
from ..mode import Mode
from ..typing import FArray

import yaml


try:
    # Use CLoader if possible, it is much faster.
    # This will make a huge difference when loading eigenvectors.
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


def load_phonons(dir_: str) -> tuple[list[Mode], list[int], list[float]]:
    """Load vibrational modes from phonopy output files.

    This function takes the directory where the files are stored and tries to detect the right file to process.

    .. seealso::

        * :func:`load_phonons_qpointsh5`
        * :func:`load_phonons_bandsh5`
        * :func:`load_phonons_qpointsyaml`
        * :func:`load_phonons_bandyaml`

    :param dir_: directory containing phonopy output files
    :return: (modes, frequencies, eigenvectors) tuple
    """
    if isfile(join(dir_, "qpoints.hdf5")):
        return load_phonons_qpointsh5(
            join(dir_, "qpoints.hdf5"), join(dir_, "phonopy.yaml")
        )

    elif isfile(join(dir_, "qpoints.hdf5.gz")):
        return load_phonons_qpointsh5(
            join(dir_, "qpoints.hdf5.gz"), join(dir_, "phonopy.yaml"), op=gzip.open
        )

    elif isfile(join(dir_, "band.hdf5")):
        return load_phonons_bandsh5(join(dir_, "band.hdf5"), join(dir_, "phonopy.yaml"))

    elif isfile(join(dir_, "band.hdf5.gz")):
        return load_phonons_bandsh5(
            join(dir_, "band.hdf5.gz"), join(dir_, "phonopy.yaml"), op=gzip.open
        )

    elif isfile(join(dir_, "qpoints.yaml")):
        return load_phonons_qpointsyaml(
            join(dir_, "qpoints.yaml"), join(dir_, "phonopy.yaml")
        )

    elif isfile(join(dir_, "band.yaml")):
        return load_phonons_bandyaml(join(dir_, "band.yaml"))

    else:
        raise FileNotFoundError("No known file to extract modes from.")


def load_phonons_bandsh5(
    bandh5: str, phyaml: str, op=open
) -> tuple[list[Mode], list[int], list[float]]:
    """Load vibrational modes from phonopy HDF5 output files.

    :param bandh5: path to band.hdf5 or band.hdf5.gz file
    :param op: (optional, open) open function, pass :func:`gzip.open` when dealing with compressed file.
    :return: (modes, frequencies, eigenvectors) tuple
    """
    struct = get_struct(phyaml)
    return _load_phonons_bandsh5(struct, bandh5, op)


def load_phonons_bandyaml(bandy: str) -> tuple[list[Mode], list[int], list[float]]:
    """Load vibrational modes from phonopy YAML output files.

    :param bandy: path to band.yaml file
    :return: (modes, frequencies, eigenvectors) tuple
    """
    with open(bandy) as f:
        raw = yaml.load(f, Loader)

    struct = PPStruct.from_yaml_cell(raw)
    return _load_phonons_bandyaml(struct, raw)


def load_phonons_qpointsh5(
    qph5: str, phyaml: str, op=open
) -> tuple[list[Mode], list[int], list[float]]:
    """Load vibrational modes from phonopy HDF5 output files.

    :param qph5: path to qpoints.hdf5 or qpoints.hdf5.gz file
    :param phyaml: path to phonopy.yaml file
    :param op: (optional, open) open function, pass :func:`gzip.open` when dealing with compressed file.
    :return: (modes, frequencies, eigenvectors) tuple
    """
    struct = get_struct(phyaml)
    return _load_phonons_qpointsh5(struct, qph5, op)


def load_phonons_qpointsyaml(
    qpyaml: str, phyaml: str
) -> tuple[list[Mode], list[int], list[float]]:
    """Load vibrational modes from phonopy YAML output files.

    :param qpyaml: path to qpoints.yaml file
    :param phyaml: path to phonopy.yaml file
    :return: (modes, frequencies, eigenvectors) tuple
    """
    struct = get_struct(phyaml)
    return _load_phonons_qpointsyaml(struct, qpyaml)


def get_struct(phyaml: str) -> PPStruct:
    "Get a structure from the phonopy.yaml file."
    if not isfile(phyaml):
        raise FileNotFoundError("Missing file phonopy.yaml")

    with open(phyaml) as f:
        raw = yaml.load(f, Loader)

    return PPStruct.from_yaml_cell(raw["supercell"])


def _load_phonons_bandsh5(
    struct: PPStruct, path: str, op
) -> tuple[list[Mode], list[int], list[float]]:
    # helper function for load_phonons_bandsh5
    import h5py

    with h5py.File(op(path, mode="rb")) as f:
        for seg in f["path"]:
            qp = np.linalg.norm(seg, axis=-1)

            (indices,) = np.where(qp == 0.0)

            if len(indices) < 1:
                i = indices[0]
                break
        else:
            raise ValueError("Only Gamma point phonons are supported.")

        # indices: segment, point, mode
        ev = f["eigenvector"][0, i, :]
        fr = f["frequency"][0, i, :]

        return _load_phonons_h5(struct, qp, ev, fr)


def _load_phonons_qpointsh5(
    struct: PPStruct, path: str, op
) -> tuple[list[Mode], list[int], list[float]]:
    # helper function for load_phonons_qpointsh5
    import h5py

    with h5py.File(op(path, mode="rb")) as f:
        qp = np.linalg.norm(f["qpoint"], axis=-1)

        (indices,) = np.where(qp == 0.0)
        if len(indices) < 1:
            raise ValueError("Only Gamma point phonons are supported.")

        i = indices[0]

        # indices: point, mode
        ev = f["eigenvector"][i, :]
        fr = f["frequency"][i, :]

        return _load_phonons_h5(struct, qp, ev, fr)


def _load_phonons_h5(
    struct: PPStruct, qp: dict, ev: list[FArray], fr: list[float]
) -> tuple[list[Mode], list[int], list[float]]:
    # helper function for _load_phonons_bandsh5 and _load_phonons_qpointsh5
    n = len(struct.atoms) * 3

    phonons = []
    for i, (v, f) in enumerate(zip(ev, fr)):
        assert v.shape == (n,), f"Wrong shape {v.shape}"
        phonons.append(
            Mode(
                struct.lattice,
                struct.atoms,
                i,
                f >= 0,
                abs(f) * THz_in_meV,
                struct.ref,
                # imaginary part can be ignored in q=G
                v.reshape((-1, 3)).real,
                struct.masses,
            )
        )

    return phonons, struct.pops, struct.masses


def _load_phonons_bandyaml(
    struct: PPStruct, raw: dict
) -> tuple[list[Mode], list[int], list[float]]:
    # helper function for load_phonons_bandyaml
    raw_ph = raw["phonon"][0]
    # TODO actually find the Gamma point
    point = raw_ph["band"]
    return _load_phonons_yaml(struct, point)


def _load_phonons_qpointsyaml(
    struct: PPStruct, path: str
) -> tuple[list[Mode], list[int], list[float]]:
    # helper function for load_phonons_qpointsyaml
    with open(path) as f:
        raw = yaml.load(f, Loader)

    raw_ph = raw["phonon"][0]

    qp = raw_ph["q-position"]
    if np.any(qp != np.array([0.0, 0.0, 0.0])):
        raise ValueError("Only Gamma point phonons are supported.")

    point = raw_ph["band"]
    return _load_phonons_yaml(struct, point)


def _load_phonons_yaml(struct, point) -> tuple[list[Mode], list[int], list[float]]:
    # helper function for _load_phonons_bandyaml and _load_phonons_qpointsyaml
    n = len(struct.atoms)
    phonons = []
    for i, ph in enumerate(point):
        f = ph["frequency"]
        # imaginary part is ignored in q=G
        # so we only take the first component of the last dimension
        v = np.array(ph["eigenvector"])[:, :, 0]

        assert v.shape == (n, 3), f"Eigenvector shape of band {i} is wrong {v.shape}."

        phonons.append(
            Mode(
                struct.lattice,
                struct.atoms,
                i,
                f >= 0,
                abs(f) * THz_in_meV,
                struct.ref,
                v,
                struct.masses,
            )
        )

    return phonons, struct.pops, struct.masses


@dataclass
class PPStruct:
    "Crystal structure as described by phonopy."
    pops: list[int]
    lattice: FArray
    masses: list[float]
    atoms: list[str]
    ref: FArray

    @classmethod
    def from_yaml_cell(cls, cell: dict) -> "PPStruct":
        """Create a PPStruct from a cell dictionary found in phonopy files."""
        lattice = np.array(cell["lattice"])
        masses = [p["mass"] for p in cell["points"]]
        atoms = [p["symbol"] for p in cell["points"]]
        ref = np.array(
            [np.array(p["coordinates"]).dot(lattice) for p in cell["points"]]
        )

        pops = [len(list(g)) for k, g in groupby(atoms)]

        return cls(pops, lattice, masses, atoms, ref)
