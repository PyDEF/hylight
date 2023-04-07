"Read vibrational modes from CRYSTAL log."
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
import numpy as np

from itertools import groupby

from ..constants import cm1_in_J, eV_in_J
from ..mode import Mode

cm1_in_meV = cm1_in_J / eV_in_J * 1000


def load_phonons(path):
    """Load phonons from a CRYSTAL17 logfile.

    :returns: :code:`(phonons, pops, masses)`

        - *phonons*: list of hylight.mode.Mode instances
        - *pops*: population for each atom species
        - *masses*: list of SI masses
    """

    phonons = []
    masses = []
    names = []

    with open(path) as log:
        for line in log:
            if "CARTESIAN COORDINATES - PRIMITIVE CELL" in line:
                break

        # *******************************************************************************
        # *      ATOM          X(ANGSTROM)         Y(ANGSTROM)         Z(ANGSTROM)
        # *******************************************************************************
        next(log)
        next(log)
        next(log)  # noqa: E702

        pos = []
        for line in log:
            l = line.strip()  # noqa: E741
            if l:
                pos.append(l.split()[3:])
            else:
                break

        pos = np.array(pos, dtype=float)

        for line in log:
            if "ATOMS ISOTOPIC MASS" in line:
                break

        next(log)

        for line in log:
            l = line.strip()  # noqa: E741
            if l:
                fields = l.split()
                masses.extend(map(float, fields[2::3]))
                names.extend(map(normalize, fields[1::3]))
            else:
                break

        _masses_12 = np.sqrt(masses).reshape((-1, 1))

        for line in log:
            if "NORMAL MODES NORMALIZED" in line:
                break

        next(log)

        head = next(log).strip()
        c = 0
        while head.startswith("FREQ"):
            freqs = map(float, head.strip().split()[1:])
            next(log)
            ats = []
            for _ in masses:
                # Drop the 13 first characters that qualify the line
                # Each line contains the infos for some number of modes
                # Note: the str to float conversion is delayed to be done
                # in batch in the next loop by numpy.
                # This is much faster than doing it here
                xs = next(log)[13:].strip().split()
                ys = next(log)[13:].strip().split()
                zs = next(log)[13:].strip().split()
                ats.append(zip(xs, ys, zs))

            # This loop actually has a small number of iteration corresponding
            # to how many columns the data is formatted in. On the other hand
            # `ats` has as many items as there are atoms in the system.
            # I am pretty sure it is fairly inneficient.
            # But, It works(TM).
            for f, disp in zip(freqs, zip(*ats)):
                c += 1  # Crystal does not index its phonons so I use a counter
                eigenvec = _masses_12 * np.array(disp, dtype=float)
                # FIXME The data found in the log file are the
                # eigendisplacement, thus I need to apply the sqrt of the mass
                # matrix to make them orthogonal (Can be verified on a system
                # with large mass differences like CaWO4). However, they are
                # still not orthonormal and I cannot find a proper explication
                # of what the amplitude is supposed to be. There is a vague
                # "classical amplitude in bohr" mention in the CRYSTAL log
                # file, but that is not very helpful in my opinion. In
                # particular it does not specify how the normalization is
                # applied, or what the amplitude depends on.
                eigenvec /= np.linalg.norm(eigenvec)

                # Note: imaginary freqs are logged as negative frequencies
                # Note 2: Mode currently expect the following units:
                # - frequency: meV
                # - positions: A
                # - eigenvec: normalized to 1
                # - masses: atomic masses
                phonons.append(
                    Mode(names, c, f >= 0, abs(f) * cm1_in_meV, pos, eigenvec, masses)
                )
            next(log)
            head = next(log).strip()

    pops = [sum(1 for _ in s) for (_, s) in groupby(names)]
    return phonons, pops, masses


def normalize(name):
    return name[0].upper() + name[1:].lower()
