"Grab forces from a collection of single point computations and compute hessian."
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
from warnings import warn
import re
from itertools import islice, dropwhile, repeat
from multiprocessing import Pool
import numpy as np
from ..utils import periodic_diff
from ..mode import project_on_asr, modes_from_dynmat
from ..constants import eV_in_J, atomic_mass


def process_phonons(
    outputs,
    ref_output,
    basis_source=None,
    amplitude=0.01,
    nproc=1,
    symm=True,
    asr_force=False,
):
    """Process a set of OUTCAR files to compute some phonons using Force based finite differences.

    :param outputs: list of OUTCAR paths corresponding to the finite displacements.
    :param ref_output: path to non displaced OUTCAR.
    :param basis_source: (optional) read a displacement basis from a path. The
        file is a npy file from numpy's save. If None, the basis is built from
        the displacements. If not None, the order of outputs *must* match the
        order of the displacements in the array.
    :param amplitude: (optional) amplitude of the displacement, only used if
        basis_source *is not* None.
    :param nproc: (optional) number of parallel processes used to load the files.
    :param symm: (optional) If True, use symmetric differences. OUTCARs *must*
        be ordered as :code:`[+delta_1, -delta_1, +delta_2, -delta_2, ...]`.
    :param asr_force: (optional, :code:`False`) enforce the accoustic sum rule
        by projecting the dynamic matrix on the subspace where homogeneous
        translation of all modes leads to a null frequency.
    :returns: the same tuple as the load_phonons functions.

    .. note::

        When using non canonical basis (displacements are not along a
        single degree of freedom of a single atom at a time) it may be important to
        provide the basis excplicity because it will avoid important rounding
        errors found in the OUTCAR.
    """
    lattice, atoms, _, pops, masses = get_ref_info(ref_output)

    n_atoms = len(atoms)
    n = 3 * n_atoms

    assert len(masses) == n_atoms

    rf, ref = get_forces_and_pos(n_atoms, ref_output)

    assert ref.shape == (n_atoms, 3)

    # canon2user is the transition matrix from canonical basis to vibrational basis from the right
    if basis_source is not None:
        canon2user = np.load(basis_source)

        if canon2user.shape != (n, n):
            raise ValueError("Basis shape and output shape are incompatible.")
    else:
        canon2user = np.zeros((n, n))

    h = np.zeros((n, n))

    data = enumerate(
        zip(
            outputs,
            repeat(n_atoms),
            repeat(lattice),
            repeat(ref),
            repeat(symm),
        )
    )

    if nproc > 1:
        # use all available process, as the columns are completely independant
        with Pool(processes=nproc) as pool:
            for i, force, delta in pool.imap_unordered(
                _extract_infos, data, max(1, len(outputs) // nproc)
            ):
                h[i, :] -= force
                if basis_source is None:
                    canon2user[i, :] += delta
    else:
        for i, force, delta in map(_extract_infos, data):
            h[i, :] -= force
            if basis_source is None:
                canon2user[i, :] += delta

    if not symm:
        h[:, :] += rf.reshape((1, -1))

    if basis_source is None:
        # Compute actual displacement and renormalize
        amplitudes = np.linalg.norm(canon2user, axis=-1).reshape((-1, 1))
        canon2user /= amplitudes
        h /= amplitudes
    elif symm:
        h /= 2 * amplitude
    else:
        h /= amplitude

    user2canon = canon2user.transpose()
    ortho = np.abs(canon2user @ user2canon - np.eye(n))

    if np.any(ortho > 1e-7):
        raise ValueError(
            f"Basis is very far from orthonormal: max error {np.max(ortho)}."
            + (
                ""
                if basis_source
                else "\nThe non orthonormality may be due to rounding errors"
                " in the atomic positions as written in VASP's outputs."
                " Try to provide the real basis with basis_source."
            )
        )
    elif np.any(ortho > 1e-12):
        warn(f"Basis does not seem orthonormal: max error {np.max(ortho)}.")

    # Rotate the matrix back to cannonical basis
    h = user2canon @ h

    # force the symetry to account for numerical imprecision
    h = 0.5 * (h + h.transpose())
    h *= eV_in_J * 1e20

    # Rotate the mass matrix too
    m12 = np.sqrt(np.diag([1.0 / (m * atomic_mass) for m in masses for _ in range(3)]))

    # dynamical matrix
    dynmat = m12.transpose() @ h @ m12

    if asr_force:
        dynmat = project_on_asr(dynmat, masses)

    return (
        modes_from_dynmat(lattice, atoms, masses, ref, dynmat),
        pops,
        masses,
        dynmat,
    )


def _extract_infos(args):
    i, (output, n_atoms, lattice, ref, symm) = args
    forces, pos = get_forces_and_pos(n_atoms, output)
    delta = periodic_diff(lattice, ref, pos)
    if symm:
        return (
            i // 2,
            (1 - 2 * (i % 2)) * forces.reshape((-1,)),
            (1 - 2 * (i % 2)) * delta.reshape((-1,)),
        )
    else:
        return (i, forces.reshape((-1,)), delta.reshape((-1,)))


def get_forces(n, path):
    """Extract n forces from an OUTCAR."""
    return _get_forces_and_pos(n, path)[:, 3:6]


def get_forces_and_pos(n, path):
    """Extract n forces and atomic positions from an OUTCAR."""
    data = _get_forces_and_pos(n, path)
    return data[:, 3:6], data[:, 0:3]


def _get_forces_and_pos(n, path):
    "Ad hoc parser for OUTCAR and vasprun.xml."

    if path.endswith(".xml"):
        with open(path) as f:
            f = dropwhile_err(
                lambda line: "<calculation>" not in line,
                f,
                ValueError("Unexpected EOF while looking for calculation."),
            )

            f = dropwhile_err(
                lambda line: '<varray name="positions"' not in line,
                f,
                ValueError("Unexpected EOF while looking for positions."),
            )

            positions = np.array(
                [line.split()[1:4] for line in islice(f, 1, n + 1)],
                dtype=float,
            )

            f = dropwhile_err(
                lambda line: '<varray name="forces"' not in line,
                f,
                ValueError("Unexpected EOF while looking for forces."),
            )

            forces = np.array(
                [line.split()[1:4] for line in islice(f, 1, n + 1)],
                dtype=float,
            )

        return np.hstack([positions, forces])

    else:  # OUTCAR
        with open(path) as outcar:
            # advance to the force block
            for line in outcar:
                if "TOTAL-FORCE (eV/Angst)" in line:
                    break
            else:
                raise ValueError("Unexpected EOF")

            # read the block and let numpy parse the numbers
            return np.array(
                [line.split() for line in islice(outcar, 1, n + 1)],
                dtype=float,
            )


def get_ref_info(path):
    """Load system infos from a OUTCAR.

    This is an ad hoc parser, so it may fail if the OUTCAR changes a lot.

    :returns: :code:`(atoms, ref, pops, masses)`

        - *atoms*: list of species names
        - *pos*: positions of atoms
        - *pops*: population for each atom species
        - *masses*: list of SI masses
    """
    if path.endswith(".xml"):
        raise ValueError("vasprun.xml is not supported for the reference file.")
    pops = []
    masses = []
    names = []
    atoms = []
    n_atoms = None
    with open(path) as outcar:
        # Extract the populations informations
        for line in outcar:
            if "VRHFIN" in line:
                line = line.strip()
                name = line.split("=")[1].strip().split(":")[0].strip()
                if name == "r":
                    name = "Zr"
                names.append(name)

            elif "ions per type" in line:
                pops = list(map(int, line.split("=")[1].split()))
                break
        else:
            raise ValueError("Unexpected EOF while looking for populations.")

        # build the atom list
        for p, n in zip(pops, names):
            atoms.extend([n] * p)

        n_atoms = len(atoms)

        outcar = dropwhile_err(
            lambda line: "POMASS" not in line,
            outcar,
            ValueError("Unexpected EOF while looking for masses."),
        )

        line = next(outcar)

        line = line.strip()
        raw = line.split("=")[1]

        # Unfortunately the format is really broken
        fmt = " " + "([ .0-9]{6})" * len(names)

        m = re.fullmatch(fmt, raw)
        assert m, "OUTCAR is not formatted as expected."

        masses = []
        for p, mass in zip(pops, m.groups()):
            masses.extend([float(mass)] * p)

        outcar = dropwhile_err(
            lambda line: line
            != "      direct lattice vectors                 reciprocal lattice vectors\n",
            outcar,
            ValueError("Unexpected EOF while looking for lattice parameters."),
        )
        next(outcar)

        data = np.array(
            [line.split() for line in islice(outcar, 0, 3)],
            dtype=float,
        )
        lattice = data[:, 0:3]

        outcar = dropwhile_err(
            lambda line: line
            != " position of ions in cartesian coordinates  (Angst):\n",
            outcar,
            ValueError("Unexpected EOF while looking for positions."),
        )
        next(outcar)

        data = np.array(
            [line.split() for line in islice(outcar, 0, n_atoms)],
            dtype=float,
        )
        pos = data[:, 0:3]

    return lattice, atoms, pos, pops, masses


def dropwhile_err(pred, it, else_err):
    "itertools.dropwhile wrapper that raise else_err if it reach the end of the file."
    rest = dropwhile(pred, it)

    try:
        first = next(rest)
    except StopIteration as e:
        raise else_err from e
    else:
        yield first

    yield from rest
