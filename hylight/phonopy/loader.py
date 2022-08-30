""" WIP

Module to load phonons frequencies and eigenvectors from phonopy output.
"""
import math
from os.path import isfile, join
from itertools import groupby
import gzip

import numpy as np

import h5py
import yaml

from ..constants import THz_in_meV
from ..mode import Mode


try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


def load_phonons(dir_):
    if isfile(join(dir_, "band.hdf5")):
        op = open
        path_struct = join(dir_, "phonopy.yaml")
        path_band = join(dir_, "band.hdf5")
    elif isfile(join(dir_, "band.hdf5.gz")):
        op = gzip.open
        path_struct = join(dir_, "phonopy.yaml")
        path_band = join(dir_, "band.hdf5.gz")
    elif isfile(join(dir_, "band.yaml")):
        # redirect to the YAML loader
        return load_phonons_yaml(join(dir_, "band.yaml"))
    else:
        raise FileNotFoundError("Missing file for band.hdf5 or band.hdf5.gz")

    if not isfile(path_struct):
        raise FileNotFoundError("Missing file phonopy.yaml")

    phonons = []

    with open(path_struct) as f:
        raw_structure = yaml.load(f, Loader)

    if raw_structure["physical_unit"]["length"] != "angstrom":
        raise ValueError(
            "Unfortunatly this loader currently support only angstrom as "
            "a length unit. Please contact the author with your usecase."
        )

    atoms = [e["symbol"] for e in raw_structure["unit_cell"]["points"]]
    masses = np.array([e["mass"] for e in raw_structure["unit_cell"]["points"]])
    lattice = np.array(raw_structure["unit_cell"]["lattice"])
    ref = np.array(
        [
            np.array(p["coordinates"]).dot(lattice)
            for p in raw_structure["unit_cell"]["points"]
        ]
    )

    pops = {k: len(list(g)) for k, g in groupby(atoms)}

    n = 3 * ref.shape[0]

    with h5py.File(op(path_band, mode="rb")) as f:
        qp = np.linalg.norm(f["path"][0], axis=-1)
        (indices,) = np.where(qp == 0.0)
        if len(indices) < 1:
            raise ValueError("Only Gamma point phonons are supported.")

        i = indices[0]

        ev = f["eigenvector"][0, i]
        fr = f["frequency"][0, i]

        for i, (v, f) in enumerate(zip(ev, fr)):
            assert v.shape == (n,), f"Wrong shape {v.shape}"
            phonons.append(
                Mode(
                    atoms,
                    i,
                    f >= 0,
                    abs(f) * THz_in_meV,
                    ref,
                    # imaginary part can be ignored in q=G
                    v.reshape((-1, 3)).real,
                    masses,
                )
            )

    return phonons, pops, masses


def load_phonons_yaml(path):
    with open(path) as f:
        raw = yaml.load(f, Loader)

    raw_ph = raw["phonon"][0]

    qp = raw_ph["q-position"]
    if np.any(qp != np.array([0.0, 0.0, 0.0])):
        raise ValueError("Only Gamma point phonons are supported.")

    lattice = np.array(raw["lattice"])
    ref = lattice.dot(np.array([
        p["coordinates"]
        for p in raw["points"]
    ]).transpose()).transpose()

    masses = [p["mass"] for p in raw["points"]]
    atoms = [p["symbol"] for p in raw["points"]]
    pops = {k: len(list(g)) for k, g in groupby(atoms)}
    n = len(atoms)

    phonons = []

    for i, ph in enumerate(raw_ph["band"]):
        f = ph["frequency"]
        # imaginary part is ignored in q=G
        v = np.array(ph["eigenvector"])[:, :, 0]

        assert v.shape == (n, 3), f"Eigenvector shape of band {i} is wrong {v.shape}."

        phonons.append(
            Mode(
                atoms,
                i,
                f >= 0,
                abs(f) * THz_in_meV,
                ref,
                v,
                masses,
            )
        )

    return phonons, pops, masses
