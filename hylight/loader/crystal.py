import numpy as np

from itertools import groupby

from ..constants import cm1_in_J, eV_in_J
from ..mode import Mode

cm1_in_meV = cm1_in_J / eV_in_J * 1000


def load_phonons(path):
    """Load phonons from a CRYSTAL17 logfile.
    :returns: (phonons, pops, masses)
      phonons: list of hylight.mode.Mode instances
      pops: population for each atom species
      masses: list of SI masses
    """

    phonons = []
    masses = []
    names = []

    with open(path) as log:
        for line in log:
            if "CARTESIAN COORDINATES" in line:
                break
        next(log)
        next(log)
        next(log)

        pos = []
        for line in log:
            if l := line.strip():
                pos.append(l.split()[3:])
            else:
                break

        pos = np.array(pos, dtype=float)

        for line in log:
            if "ATOMS ISOTOPIC MASS" in line:
                break

        next(log)

        for line in log:
            if l := line.strip():
                fields = l.split()
                masses.extend(map(float, fields[2::3]))
                names.extend(map(normalize, fields[1::3]))
            else:
                break

        for line in log:
            if "NORMAL MODES NORMALIZED" in line:
                break

        next(log)

        head = next(log).strip()
        c = 0
        while "FREQ" in head:
            freqs = map(float, head.strip().split()[1:])
            next(log)
            ats = []
            for _ in masses:
                # Drop the 13 firs characters that qualify the line
                # Each line contains the infos for some number of modes
                # Note: the str to float conversion is delayed to be done
                # in batch in the next loop by numpy.
                # This is much faster than doing it here
                xs = next(log)[13:].strip().split()
                ys = next(log)[13:].strip().split()
                zs = next(log)[13:].strip().split()
                ats.append(zip(xs, ys, zs))

            # This loop actually has a small number of iteration corresponding
            # to how many columns the data is formatted in
            # However ats has as many items as there are atoms in the system
            # I am not sure that approach is efficient
            # Yet, It works(TM).
            for f, disp in zip(freqs, zip(*ats)):
                c += 1  # Crystal does not index its phonons so I use a counter
                delta = np.array(disp, dtype=float)
                delta /= np.linalg.norm(delta)

                # Note: imaginary freqs are logged as negative frequencies
                # Note 2: Mode currently expect the following units:
                # - frequency: meV
                # - positions: A
                # - delta: normalized to 1
                # - masses: atomic masses
                phonons.append(Mode(
                    names, c, f >= 0, abs(f) * cm1_in_meV, pos, delta, masses
                ))
            next(log)
            head = next(log)

    pops = [sum(1 for _ in s) for (_, s) in groupby(names)]
    return phonons, pops, masses


def normalize(name):
    return name[0].upper() + name[1:].lower()
