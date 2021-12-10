import re

import numpy as np

from pydef.core.vasp import Poscar

from .mode import Mode


head_re = re.compile(r"^(\d+) (f  |f/i)= *.* (\d+\.\d+) meV")
mass_re = re.compile(r"^\s*POMASS\s+=\s+(\d+\.\d+)\s*;.*$")


def load_phonons(path, bias=40):
    phonons = []
    pops = []
    masses = []
    with open(path) as outcar:
        while True:
            line = outcar.readline()
            if not line:
                break
            line = line.strip()
            if "ions per type" in line:
                pops = map(int, line.split("=")[1].strip().split())
            elif "POMASS" in line and not "ZVAL" in line:
                masses_ = map(float, line.split("=")[1].strip().split())
                masses = []

                for p, m in zip(pops, masses_):
                    masses.extend([m] * p)
            elif m := head_re.fullmatch(line):
                n, im, ener = m.groups()

                if float(ener) < bias:
                    continue

                outcar.readline()

                data = np.array([outcar.readline().strip().split() for _ in range(40)], dtype=float)

                ref = data[:, 0:3]
                delta = data[:, 3:6]

                phonons.append(Mode(n, im.strip() == "f", float(ener), ref, delta, masses))
    return phonons, pops, masses


def load_poscar(path):
    return Poscar.from_file(path).raw
