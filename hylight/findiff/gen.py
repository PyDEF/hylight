"""Generate a collection of positions for finite differences computations.
"""
# Copyright (c) 2024, Théo Cavignac <theo.cavignac+dev@gmail.com>, The PyDEF team <camille.latouche@cnrs-imn.fr>
# Licensed under the EUPL
import os.path

import numpy as np
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
from ..vasp.common import Poscar
from ..mode import orthonormalize


def save_random_disp(
    source,
    disp_dest=".",
    basis_dest="basis.npy",
    amplitude=0.01,
    seed=0,
    symm=False,
):
    """Produce displaced POSCARs in random directions.

    Takes a reference position and displaces it into random directions to
    produce a set of positions for finite differences computations.
    It also save the set of displacements in basis_dest.

    :param source: reference POSCAR
    :param disp_dest: (optional, ".") directory name or callable that determine
        the destination of the POSCARs.
    :param basis_dest: (optional, "basis.npy") Name of the basis set file. It
        is a npy file made with numpy's save function.
    :param amplitude: (optional, 0.01) amplitude of the displacements in Angstrom.
    :param seed: (optional, 0) random seed to use to generate the displacements.
    :param symm: (optional, False) when True, generate both ref+delta and
        ref-delta POSCARs for each direction.
    """
    ref = Poscar.from_file(source)
    n, _ = ref.raw.shape
    basis = random_basis(3 * n, seed=seed)

    np.save(basis_dest, basis)

    save_disp(ref, basis, disp_dest=disp_dest, amplitude=amplitude, symm=symm)


def save_disp_from_basis(
    source,
    basis_source,
    disp_dest=".",
    amplitude=0.01,
    symm=False,
):
    """Produce displaced POSCARs from a given set of directions.

    Takes a reference position and displaces it into directions from
    basis_source to produce a set of positions for finite differences
    computations.

    :param source: reference POSCAR
    :param basis_source: Name of the basis set file. It is a npy file made with
        numpy's save function. Each row is a direction.
    :param disp_dest: (optional, ".") directory name or callable that determine
        the destination of the POSCARs.
    :param amplitude: (optional, 0.01) amplitude of the displacements in Angstrom.
    :param symm: (optional, False) when True, generate both ref+delta and
        ref-delta POSCARs for each direction.
    """
    ref = Poscar.from_file(source)
    n, _ = ref.raw.shape
    basis = np.load(basis_source)

    save_disp(ref, basis, disp_dest=disp_dest, amplitude=amplitude, symm=symm)


def save_disp(ref, basis, disp_dest=".", amplitude=0.01, symm=False):
    """Produce displaced POSCARs from a given set of directions.

    Takes a reference position and displaces it into directions from
    basis to produce a set of positions for finite differences
    computations.

    :param ref: reference Poscar instancce
    :param basis: nupy array where each row is a direction.
    :param disp_dest: (optional, ".") directory name or callable that determine
        the destination of the POSCARs.
    :param amplitude: (optional, 0.01) amplitude of the displacements in Angstrom.
    :param symm: (optional, False) when True, generate both ref+delta and
        ref-delta POSCARs for each direction.
    """
    assert isinstance(ref, Poscar)
    assert ref.raw.shape[0] * 3 == basis.shape[0] == basis.shape[1]

    if callable(disp_dest):
        for i, d in enumerate(gen_disp(ref, basis, amplitude=amplitude, symm=symm)):
            d.to_file(disp_dest(i))
    else:
        for i, d in enumerate(gen_disp(ref, basis, amplitude=amplitude, symm=symm)):
            d.to_file(os.path.join(disp_dest, f"POSCAR-{i:03}"))


def gen_disp(ref, basis, amplitude=0.01, symm=False):
    """Iterate over displaced Poscar instances from a given set of directions.

    Takes a reference position and displaces it into directions from
    basis to produce a set of positions for finite differences
    computations.

    :param ref: reference Poscar instancce
    :param basis: nupy array where each row is a direction.
    :param amplitude: (optional, 0.01) amplitude of the displacements in Angstrom.
    :param symm: (optional, False) when True, generate both ref+delta and
        ref-delta POSCARs for each direction.
    """
    assert isinstance(ref, Poscar)
    n, _ = ref.raw.shape

    if basis.shape != (3 * n, 3 * n):
        raise ValueError(f"Invalid basis size. Expected shape ({3 * n}, {3 * n}).")

    cp = ref.copy()

    for i in range(3 * n):
        cp.raw = ref.raw + (amplitude * basis[i, :]).reshape((n, 3))

        yield cp

        if symm:
            cp.raw = ref.raw - (amplitude * basis[i, :]).reshape((n, 3))

            yield cp


def random_basis(n, seed=0):
    """Generate a random basis matrix or rank n.

    :param n: Rank of the basis
    :param seed: (optional, 0) randomness seed.
    :returns: a (n, n) orthonormal numpy array.
    """
    mat = RandomState(MT19937(SeedSequence(seed))).rand(n, n)
    orthonormalize(mat)
    return mat
