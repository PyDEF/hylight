"Common utilities to read CRYSTAL files."
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

from ..utils import InputError
from ..struct import Struct


class CrystalOut(Struct):
    @classmethod
    def from_file(cls, filename):
        """Load a POSCAR file

        :param filename: path to the file to read
        :returns: a Poscar object.
        """

        with open(filename) as f:
            for line in f:
                if "DIRECT LATTICE VECTORS CARTESIAN" in line:
                    break

            try:
                next(f)
            except StopIteration as e:
                raise InputError("Unexpected end of file.") from e

            try:
                cell = np.array(
                    [next(f).split(), next(f).split(), next(f).split()], dtype=float
                )
            except StopIteration as e:
                raise InputError("Unexpected end of file.") from e
            except ValueError as e:
                raise InputError("Invalid line in cell parameters block.") from e

            for line in f:
                if "CARTESIAN COORDINATES - PRIMITIVE CELL" in line:
                    break

            # *******************************************************************************
            # *      ATOM          X(ANGSTROM)         Y(ANGSTROM)         Z(ANGSTROM)
            # *******************************************************************************
            try:
                next(f)
                next(f)
                next(f)
            except StopIteration as e:
                raise InputError("Unexpected end of file.") from e

            pos = {}
            for line in f:
                l = line.strip()  # noqa: E741
                if l:
                    sp, x, y, z = l.split()[2:]
                    if len(sp) == 2:
                        sp = sp[0] + sp[1].lower()
                    pos.setdefault(sp, []).append((x, y, z))
                else:
                    break

            return cls(
                cell, {sp: np.array(lst, dtype=float) for sp, lst in pos.items()}
            )
