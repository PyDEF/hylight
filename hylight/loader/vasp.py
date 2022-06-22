import re
from itertools import islice

import numpy as np

from pydef.core.vasp import Poscar

from ..mode import Mode


mass_re = re.compile(r"^\s*POMASS\s+=\s+(\d+\.\d+)\s*;.*$")
head_re = re.compile(r"^(\d+) (f  |f/i)= *.* (\d+\.\d+) meV")


def load_phonons(path):
    """Load phonons from a OUTCAR.
    :returns: (phonons, pops, masses)
      phonons: list of hylight.mode.Mode instances
      pops: population for each atom species
      masses: list of SI masses
    """
    phonons = []
    pops = []
    masses = []
    names = []
    atoms = []
    n_atoms = None
    n_modes = 0
    with open(path) as outcar:
        for line in outcar:
            line = line.strip()
            if "VRHFIN" in line:
                name = line.split("=")[1].strip().split(":")[0].strip()
                if name == "r":
                    name = "Zr"
                names.append(name)

            elif "ions per type" in line:
                pops = list(map(int, line.split("=")[1].strip().split()))
                break
        else:
            raise ValueError("Unexpected EOF")

        for p, n in zip(pops, names):
            atoms.extend([n] * p)

        n_atoms = len(atoms)

        for line in outcar:
            line = line.strip()
            if "POMASS" in line:
                raw = line.split("=")[1]

                # Unfortunately the format is really broken
                fmt = " " + "([ .0-9]{6})" * len(names)

                m = re.fullmatch(fmt, raw)
                assert m, "OUTCAR is not formatted as expected."

                masses = []
                for p, m in zip(pops, m.groups()):
                    masses.extend([float(m)] * p)
                break
        else:
            raise ValueError("Unexpected EOF")

        for line in outcar:
            line = line.strip()
            if "THz" in line and (m := head_re.fullmatch(line)):
                n, im, ener = m.groups()

                data = np.array(
                    [line.strip().split() for line in islice(outcar, 1, n_atoms + 1)],
                    dtype=float,
                )
                ref = data[:, 0:3]
                delta = data[:, 3:6]

                phonons.append(
                    Mode(atoms, n, im.strip() == "f", float(ener), ref, delta, masses)
                )

                n_modes += 1
                if n_modes >= 3 * n_atoms:
                    break

    return phonons, pops, masses


def load_poscar(path):
    """Read the positions from a POSCAR.

    :returns: a np.ndarray((natoms, 3), dtype=float)
    """
    return Poscar.from_file(path).raw


def load_poscar_latt(path):
    """Read the positions and the lattice parameters from a POSCAR.

    :returns: a (np.ndarray((natoms, 3), dtype=float), nd.array((3, 3), dtype=float))
      first element is the set of positions
      second element is the lattice parameters
    """
    p = Poscar.from_file(path)
    return p.raw, p.cell_parameters
