"Pervasive utilities for hylight.vasp submodule."
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
from copy import deepcopy as copy

import numpy as np

from ..constants import atomic_mass
from ..multi_phonons import compute_delta_R
from ..mode import get_energies, Mask

from ..loader import load_phonons
from .common import Poscar


def make_finite_diff_poscar(
    outcar,
    poscar_gs,
    poscar_es,
    A=0.01,
    load_phonons=load_phonons,
    bias=0,
    mask=None,
):
    """Compute positions for evaluation of the curvature of the ES PES.

    :param outcar: the name of the file where the phonons will be read.
    :param poscar_gs: the path to the ground state POSCAR
    :param poscar_es: the path to the excited state POSCAR, it will be used as
        a base for the generated Poscars
    :param A: (optional, 0.01) the amplitude of the displacement in A.
    :param load_phonons: (optional, vasp.loader.load_phonons) the procedure use
        to read outcar
    :return: :code:`(mu, pes_left, pes_right)`

        - mu: the effective mass in kg
        - pes_left: a Poscar instance representing the left displacement
        - pes_right: a Poscar instance representing the right displacement
    """

    if mask is None:
        mask = Mask.from_bias(bias)

    delta_R = compute_delta_R(poscar_gs, poscar_es)
    phonons, _, masses = load_phonons(outcar)
    phonons = [p for p in phonons if p.real]

    m = np.array(masses).reshape((-1, 1))
    delta_Q = np.sqrt(m) * delta_R

    k = get_energies(phonons, mask=mask) ** 2
    d = np.array(
        [np.sum(p.eigenvector * delta_Q) for p in phonons if mask.accept(p.energy)]
    )

    kd = k * d

    grad = m ** (-0.5) * np.sum(
        (np.array([p.eigenvector for p in phonons]) * kd.reshape((-1, 1, 1))), axis=0
    )

    g_dir = -grad / np.linalg.norm(grad)

    delta = A * g_dir

    pes = Poscar.from_file(poscar_es)
    pes_left = copy(pes)
    pes_right = copy(pes)

    pes_left.raw -= delta
    pes_right.raw += delta

    mu = (np.linalg.norm(g_dir, axis=-1) ** 2).dot(masses) * atomic_mass
    return mu, pes_left, pes_right
