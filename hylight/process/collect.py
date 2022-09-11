import re
from itertools import islice
import numpy as np
from ..mode import Mode
from ..constants import hbar_si, eV_in_J


def process_phonons(basis_source, ref_output, outputs, amplitude=0.01):
    basis = np.load(basis_source)

    n = basis.shape[0]
    n_atom = n // 3

    bt = basis.transpose()

    rf = get_forces(n_atom, ref_output).reshape((-1,))

    atoms, ref, pops, masses = get_ref_info(ref_output)

    if rf.shape != (n_atom, 3):
        raise ValueError("Basis shape and output shape are incompatible.")

    h = np.ndarray((n, n))

    for i, output in enumerate(outputs):
        h[i, :] = get_forces(n_atom, output).reshape((-1,))

    h[:, :] -= rf.reshape((-1, 1))

    h /= amplitude

    # FIXME check that this make sense
    h = h @ basis

    # FIXME check that bt and basis are in the right order
    sqrt_m = bt @ np.sqrt(np.diag([m for m in masses for _ in range(3)])) @ basis

    dynmat = sqrt_m.transpose() @ h @ sqrt_m

    vals, vecs = np.linalg.eig(dynmat)
    energies = hbar_si * np.sqrt(np.abs(vals)) / eV_in_J * 1e3

    vx = bt @ vecs

    return (
        [
            Mode(
                atoms,
                n,
                e2 < 0,
                e,
                ref,
                vx.reshape((-1, 3)),
                masses,
            )
            for e2, e, v in zip(vals, energies, vx)
        ],
        pops,
        masses,
    )


def get_forces(n, path):
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

        forces = data[:, 3:6]

    return forces


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
            raise ValueError("Unexpected EOF")

        for p, n in zip(pops, names):
            atoms.extend([n] * p)

        n_atoms = len(atoms)

        for line in outcar:
            if "POMASS" in line:
                line = line.strip()
                raw = line.split("=")[1]

                # Unfortunately the format is really broken
                fmt = " " + "([ .0-9]{6})" * len(names)

                m = re.fullmatch(fmt, raw)
                assert m, "OUTCAR is not formatted as expected."

                masses = []
                for p, mass in zip(pops, m.groups()):
                    masses.extend([float(mass)] * p)
                break
        else:
            raise ValueError("Unexpected EOF")

        for line in outcar:
            if line == " position of ions in fractional coordinates  (Angst):\n":
                break
        else:
            raise ValueError("Unexpected EOF")

        data = np.array(
            [line.split() for line in islice(outcar, 0, n_atoms)],
            dtype=float,
        )
        pos = data[:, 0:3]

    return atoms, ref, pops, masses
