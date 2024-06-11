"""Common utilities to read CRYSTAL files.
"""

# Copyright (c) 2024, Théo Cavignac <theo.cavignac+dev@gmail.com>, The PyDEF team <camille.latouche@cnrs-imn.fr>
# Licensed under the EUPL
import numpy as np

from ..utils import InputError
from ..struct import Struct


class CrystalOut(Struct):
    "Struct from a CRYSTAL output file."

    @classmethod
    def from_file(cls, filename):
        """Read a structure from a CRYSTAL log file.

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
