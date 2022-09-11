import os.path

import numpy as np
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
from ..vasp.common import Poscar


def save_random_disp(
    source, disp_dest=".", basis_dest="basis.npy", amplitude=0.01, seed=0
):
    ref = Poscar.from_file(source)
    n, _ = ref.raw.shape
    basis = random_basis(3 * n, seed=seed)

    np.save(basis_dest, basis)

    save_disp(ref, basis, disp_dest=disp_dest, amplitude=amplitude)


def save_disp_from_basis(
    source, basis_source, disp_dest=".", amplitude=0.01
):
    ref = Poscar.from_file(source)
    n, _ = ref.raw.shape
    basis = np.load(basis_source)

    save_disp(ref, basis, disp_dest=disp_dest, amplitude=amplitude)


def save_disp(ref, basis, disp_dest=".", amplitude=0.01):
    assert isinstance(ref, Poscar)
    assert ref.shape[0] * 3 == basis.shape[0] == basis.shape[1]

    if callable(disp_dest):
        for i, d in enumerate(gen_disp(ref, basis, amplitude=amplitude)):
            d.to_file(disp_dest(i))
    else:
        for i, d in enumerate(gen_disp(ref, basis, amplitude=amplitude)):
            d.to_file(os.path.join(disp_dest, f"POSCAR-{i:03}"))


def gen_disp(ref, basis, amplitude=0.01):
    assert isinstance(ref, Poscar)
    n, _ = ref.raw.shape

    if basis.shape != (3 * n, 3 * n):
        raise ValueError(f"Invalid basis size. Expected shape ({3 * n}, {3 * n}).")

    cp = ref.copy()

    for i in range(3 * n):
        cp.raw = ref.raw + (amplitude * basis[i, :]).reshape((n, 3))

        yield cp


def random_basis(n, seed=0):
    rs = RandomState(MT19937(SeedSequence(seed)))
    q, _ = np.linalg.qr(rs.rand(n, n))
    return q
