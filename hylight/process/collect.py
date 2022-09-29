import re
from itertools import islice, dropwhile, repeat
from multiprocessing import Pool
import numpy as np
from ..utils import gen_translat
from ..mode import Mode
from ..constants import hbar_si, eV_in_J, atomic_mass


def process_phonons(outputs, ref_output, basis_source=None, amplitude=0.01, nproc=1):
    """Process a set of OUTCAR files to compute some phonons using Force based finite differences.

    :param outputs: list of OUTCAR paths corresponding to the finite displacements.
    :param ref_output: path to non displaced OUTCAR.
    :param basis_source: read a displacement basis from a path. If None, the basis is built from the displacements.
    :param amplitude: amplitude of the displacement, only use if basis_source is not None.
    :param nproc: number of parallel processes used to load the files.
    """
    lattice, atoms, _, pops, masses = get_ref_info(ref_output)

    n_atoms = len(atoms)
    n = 3 * n_atoms

    assert len(masses) == n_atoms

    rf, ref = get_forces_and_pos(n_atoms, ref_output)

    assert ref.shape == (n_atoms, 3)

    # basis is the transition matrix from canonic to vibration from the right
    if basis_source is not None:
        basis = np.load(basis_source)

        if basis.shape != (n, n):
            raise ValueError("Basis shape and output shape are incompatible.")
    else:
        basis = np.ndarray((n, n))

    h = np.ndarray((n, n))

    data = enumerate(
        zip(
            outputs,
            repeat(n_atoms),
            repeat(lattice),
            repeat(ref),
        )
    )

    if nproc > 1:
        # use all available process, as the columns are completely independant
        with Pool(processes=nproc) as pool:
            for i, force, delta in pool.imap_unordered(
                extract_infos, data, max(1, len(outputs) // nproc)
            ):
                h[i, :] = -force
                if basis_source is None:
                    basis[i, :] = delta
    else:
        for i, force, delta in map(extract_infos, data):
            h[i, :] = -force
            if basis_source is None:
                basis[i, :] = delta

    h[:, :] += rf.reshape((1, -1))

    if basis_source is None:
        # Compute actual displacement and renormalize
        amplitudes = np.linalg.norm(basis, axis=-1).reshape((-1, 1))
        basis /= amplitudes
        h /= amplitudes
    else:
        h /= amplitude

    bt = basis.transpose()

    # Rotate the matrix to get it into the right basis
    h = basis @ h

    # force the symetry to account for numerical imprecision
    h = 0.5 * (h + h.transpose())
    h *= eV_in_J * 1e20

    assert np.all(basis @ bt - np.eye(n) < 1e-15), "basis is not orthogonal"

    # Rotate the mass matrix too
    m = basis @ np.sqrt(np.diag([1. / (m * atomic_mass) for m in masses for _ in range(3)])) @ bt

    dynmat = m.transpose() @ h @ m

    vals, vecs = np.linalg.eig(dynmat)
    assert vecs.shape == (n, n)

    energies = hbar_si * np.sqrt(np.abs(vals)) / eV_in_J * 1e3

    vx = (bt @ vecs).reshape((n, n_atoms, 3))

    return (
        [
            Mode(
                atoms,
                n,
                e2 >= 0,
                e,
                ref,
                v.reshape((-1, 3)),
                masses,
            )
            for e2, e, v in zip(vals, energies, vx)
        ],
        pops,
        masses,
    )


def extract_infos(args):
    i, (output, n_atoms, lattice, ref) = args
    forces, pos = get_forces_and_pos(n_atoms, output)
    delta = compute_delta(lattice, ref, pos)
    return (i, forces.reshape((-1,)), delta.reshape((-1,)))


def get_forces(n, path):
    """Extract n forces from an OUTCAR."""
    return _get_forces_and_pos(n, path)[:, 3:6]


def get_forces_and_pos(n, path):
    """Extract n forces and atomic positions from an OUTCAR."""
    e, data = _get_forces_and_pos(n, path)
    return e, data[:, 3:6], data[:, 0:3]


def _get_forces_and_pos(n, path):
    with open(path) as outcar:

        for line in outcar:
            if "TOTAL-FORCE (eV/Angst)" in line:
                break
        else:
            raise ValueError("Unexpected EOF")

        data = np.array(
            [line.split() for line in islice(outcar, 1, n + 1)],
            dtype=float,
        )

    return data


def get_ref_info(path):
    """Load system infos from a OUTCAR.

    :returns: (atoms, ref, pops, masses)
      atoms: list of species names
      pos: positions of atoms
      pops: population for each atom species
      masses: list of SI masses
    """
    pops = []
    masses = []
    names = []
    atoms = []
    n_atoms = None
    with open(path) as outcar:
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

        for p, n in zip(pops, names):
            atoms.extend([n] * p)

        n_atoms = len(atoms)

        outcar = drop_while_err(
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

        outcar = drop_while_err(
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

        outcar = drop_while_err(
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


def compute_delta(lattice, ref, disp):
    dp = disp - ref
    d = np.array([dp - t for t in gen_translat(lattice)])  # shape (27, n, 3)

    # norms is an array of length of delta
    norms = np.linalg.norm(d, axis=2)  # shape = (27, n)
    best_translat = np.argmin(norms, axis=0)  # shape = (n,)

    n = d.shape[1]
    res = np.ndarray((n, 3))

    for i, k in enumerate(best_translat):
        res[i, :] = d[k, i, :]

    return res


def skip_until(file, ref, descr):
    for line in outcar:
        if line == ref:
            return

    raise ValueError("Unexpected EOF while looking for {descr}.")


def drop_while_err(pred, it, else_err):
    rest = dropwhile(pred, it)

    try:
        first = next(rest)
    except StopIteration:
        raise else_err
    else:
        yield first

    yield from rest
