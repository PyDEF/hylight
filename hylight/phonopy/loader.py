"""Module to load phonons frequencies and eigenvectors from phonopy output files.

It always uses PyYAML, but it can also need h5py to read hdf5 files.
"""
import math
from os.path import isfile, join
from itertools import groupby
import gzip
from dataclasses import dataclass

import numpy as np

from ..constants import THz_in_meV
from ..mode import Mode

import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


try:
    l = list[int]
except TypeError:
    from typing import List, Dict
else:
    List = list
    Dict = dict

try:
    from numpy.typing import NDArray
except ImportError:
    from typing import Generic, TypeVar

    T = TypeVar("T")

    class NDArray(Generic[T]):
        pass


def load_phonons(dir_):
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


def load_phonons_bandsh5(bandh5, phyaml, op=open):
    struct = get_struct(phyaml)
    return _load_phonons_bandsh5(struct, bandh5, op)


def load_phonons_bandyaml(bandy):
    with open(bandy) as f:
        raw = yaml.load(f, Loader)

    struct = Struct.from_yaml_cell(raw)
    return _load_phonons_bandyaml(struct, raw)


def load_phonons_qpointsh5(qph5, phyaml, op=open):
    struct = get_struct(phyaml)
    return _load_phonons_qpointsh5(struct, qph5, op)


def load_phonons_qpointsyaml(qpyaml, phyaml):
    struct = get_struct(phyaml)
    return _load_phonons_qpointsyaml(struct, qpyaml)


def get_struct(phyaml):
    if not isfile(phyaml):
        raise FileNotFoundError("Missing file phonopy.yaml")

    with open(phyaml) as f:
        raw = yaml.load(f, Loader)

    return Struct.from_yaml_cell(raw["supercell"])


def _load_phonons_bandsh5(struct, path, op):
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


def _load_phonons_qpointsh5(struct, path, op):
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


def _load_phonons_h5(struct, qp, ev, fr):
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


def _load_phonons_bandyaml(struct, raw):
    raw_ph = raw["phonon"][0]
    # TODO actually find the Gamma point
    point = raw_ph["band"]
    return _load_phonons_yaml(struct, point)


def _load_phonons_qpointsyaml(struct, path):
    with open(path) as f:
        raw = yaml.load(f, Loader)

    raw_ph = raw["phonon"][0]

    qp = raw_ph["q-position"]
    if np.any(qp != np.array([0.0, 0.0, 0.0])):
        raise ValueError("Only Gamma point phonons are supported.")

    point = raw_ph["band"]
    return _load_phonons_yaml(struct, point)


def _load_phonons_yaml(struct, point):
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
    pops: Dict[str, int]
    lattice: NDArray[float]
    masses: List[float]
    atoms: List[str]
    ref: NDArray[float]

    @classmethod
    def from_yaml_cell(cls, cell):
        lattice = np.array(cell["lattice"])
        masses = [p["mass"] for p in cell["points"]]
        atoms = [p["symbol"] for p in cell["points"]]
        ref = np.array(
            [np.array(p["coordinates"]).dot(lattice) for p in cell["points"]]
        )

        pops = [len(list(g)) for k, g in groupby(atoms)]

        return cls(pops, lattice, masses, atoms, ref)
