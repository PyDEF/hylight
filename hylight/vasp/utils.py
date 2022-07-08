from copy import deepcopy as copy

import numpy as np

from pydef.core.vasp import Poscar

from ..constants import atomic_mass
from .loader import load_phonons
from ..multi_phonons import compute_delta_R


def make_finite_diff_poscar(
    outcar, poscar_gs, poscar_es, A=0.01, load_phonons=load_phonons
):
    """Compute positions for evaluation of the curvature of the ES PES.

    :param outcar: the name of the file where the phonons will be read.
    :param poscar_gs: the path to the ground state POSCAR
    :param poscar_es: the path to the excited state POSCAR, it will be used as
      a base for the generated Poscars
    :param A: (optional, 0.01) the amplitude of the displacement in A.
    :param load_phonons: (optional, vasp.loader.load_phonons) the procedure use
      to read outcar
    :return: (mu, pes_left, pes_right)
      mu: the effective mass in kg
      pes_left: a Poscar instance representing the left displacement
      pes_right: a Poscar instance representing the right displacement
    """
    delta_R = compute_delta_R(poscar_gs, poscar_es)
    phonons, _, masses = load_phonons(outcar)

    k = np.array([p.energy**2 for p in phonons])
    d = np.array([p.project_coef2(delta_R) ** 0.5 for p in phonons])

    grad = k * d

    g_dir = -grad / np.linalg.norm(grad)

    delta = A * np.sum(
        np.array([p.delta for p in phonons]) * g_dir.reshape((-1, 1, 1)), axis=0
    )

    pes = Poscar.from_file(poscar_es)
    pes_left = copy(pes)
    pes_right = copy(pes)

    pes_left.raw -= delta
    pes_right.raw += delta

    mu = (np.linalg.norm(delta, axis=-1) ** 2 / A**2).dot(masses) * atomic_mass
    return mu, pes_left, pes_right
