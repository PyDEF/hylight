import re
from itertools import islice

import numpy as np

from .common import Poscar

from ..mode import Mode


mass_re = re.compile(r"^\s*POMASS\s+=\s+(\d+\.\d+)\s*;.*$")
head_re = re.compile(r"^([ 0-9]{4}) (f  |f/i)= *.* (\d+\.\d+) meV\s")


def load_phonons(path):
    """Load phonons from a OUTCAR.

    Remark: This function is a bit heavy because of text parsing. You may want
    to use hyligh-pkl to parse one for all the file and later load that
    preparsed file using pkl.load_phonons instead.

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
                for p, m in zip(pops, m.groups()):
                    masses.extend([float(m)] * p)
                break
        else:
            raise ValueError("Unexpected EOF")

        for line in outcar:
            m = head_re.fullmatch(line)
            if "THz" in line and m:
                line = line.strip()
                n, im, ener = m.groups()

                data = np.array(
                    [line.split() for line in islice(outcar, 1, n_atoms + 1)],
                    dtype=float,
                )
                ref = data[:, 0:3]
                eigenv = data[:, 3:6]

                phonons.append(
                    Mode(atoms, n, im == "f  ", float(ener), ref, eigenv, masses)
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
