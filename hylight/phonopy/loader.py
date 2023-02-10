"""Module to load phonons frequencies and eigenvectors from phonopy output files.

It always uses PyYAML, but it can also need h5py to read hdf5 files.
"""
# License:
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
from __future__ import annotations

from os.path import isfile, join
from itertools import groupby
import gzip
from dataclasses import dataclass

import numpy as np

from ..constants import THz_in_meV
from ..mode import Mode

import yaml

try:  # Use CLoader if possible, it is much faster
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


try:  # numpy.typing is a feature of Numpy 1.20
    from numpy.typing import NDArray
except ImportError:
    from typing import Generic, TypeVar

    T = TypeVar("T")

    class NDArray(Generic[T]):
        pass


def load_phonons(dir_: str) -> tuple[list[Mode], list[int], list[float]]:
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
    struct = get_struct(phyaml)
    return _load_phonons_bandsh5(struct, bandh5, op)


def load_phonons_bandyaml(bandy: str) -> tuple[list[Mode], list[int], list[float]]:
    with open(bandy) as f:
        raw = yaml.load(f, Loader)

    struct = Struct.from_yaml_cell(raw)
    return _load_phonons_bandyaml(struct, raw)


def load_phonons_qpointsh5(
    qph5: str, phyaml: str, op=open
) -> tuple[list[Mode], list[int], list[float]]:
    struct = get_struct(phyaml)
    return _load_phonons_qpointsh5(struct, qph5, op)


def load_phonons_qpointsyaml(
    qpyaml: str, phyaml: str
) -> tuple[list[Mode], list[int], list[float]]:
    struct = get_struct(phyaml)
    return _load_phonons_qpointsyaml(struct, qpyaml)


def get_struct(phyaml: str) -> Struct:
    if not isfile(phyaml):
        raise FileNotFoundError("Missing file phonopy.yaml")

    with open(phyaml) as f:
        raw = yaml.load(f, Loader)

    return Struct.from_yaml_cell(raw["supercell"])


def _load_phonons_bandsh5(
    struct: Struct, path: str, op
) -> tuple[list[Mode], list[int], list[float]]:
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
    struct: Struct, path: str, op
) -> tuple[list[Mode], list[int], list[float]]:
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
    struct: Struct, qp: dict, ev: list[NDArray], fr: list[float]
) -> tuple[list[Mode], list[int], list[float]]:
    n = len(struct.atoms) * 3

    phonons = []
    for i, (v, f) in enumerate(zip(ev, fr)):
        assert v.shape == (n,), f"Wrong shape {v.shape}"
        phonons.append(
            Mode(
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
    struct: Struct, raw: dict
) -> tuple[list[Mode], list[int], list[float]]:
    raw_ph = raw["phonon"][0]
    # TODO actually find the Gamma point
    point = raw_ph["band"]
    return _load_phonons_yaml(struct, point)


def _load_phonons_qpointsyaml(
    struct: Struct, path: str
) -> tuple[list[Mode], list[int], list[float]]:
    with open(path) as f:
        raw = yaml.load(f, Loader)

    raw_ph = raw["phonon"][0]

    qp = raw_ph["q-position"]
    if np.any(qp != np.array([0.0, 0.0, 0.0])):
        raise ValueError("Only Gamma point phonons are supported.")

    point = raw_ph["band"]
    return _load_phonons_yaml(struct, point)


def _load_phonons_yaml(struct, point) -> tuple[list[Mode], list[int], list[float]]:
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
class Struct:
    pops: list[int]
    lattice: NDArray[float]
    masses: list[float]
    atoms: list[str]
    ref: NDArray[float]

    @classmethod
    def from_yaml_cell(cls, cell: dict) -> "Struct":
        lattice = np.array(cell["lattice"])
        masses = [p["mass"] for p in cell["points"]]
        atoms = [p["symbol"] for p in cell["points"]]
        ref = np.array(
            [np.array(p["coordinates"]).dot(lattice) for p in cell["points"]]
        )

        pops = [len(list(g)) for k, g in groupby(atoms)]

        return cls(pops, lattice, masses, atoms, ref)
