"A generic representation of a crystal cell."
# License
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

from .constants import electronegativity


class Struct:
    """A general description of a periodic crystal cell.

    Store all the required infos to describe a given set of atomic positions.
    """

    def __init__(self, lattice, species, species_names=None):
        """Store all informations about a unit cell.

        See also :py:class:`hylight.vasp.common.Poscar`.

        :param lattice: a 3x3 np.array with lattice vectors in line
        :param species: a `dict[str, numpy.ndarray]` where the key is the name of the
          species and the array list positions.

          .. warning::

            Positions are in cartesian representation, not in fractional
            representation. Unit is Angstrom.
        :param species_names: (optional, None) list of species names specifying in wich order
          the positions are found in the raw representation.
        """
        self.lattice = lattice
        self.species = species
        self._system_name = None
        if species_names is None:
            self._species_names = sorted(
                self.species.keys(), key=lambda p: electronegativity[p]
            )
        else:
            self._species_names = list(species_names)
            assert len(self._species_names) == len(self.species)
            assert set(self._species_names) == set(self.species.keys())

    @classmethod
    def from_mode(cls, mode):
        positions = mode.ref
        lattice = mode.lattice

        d = {}
        names = []

        for i, sp in enumerate(mode.atoms):
            d.setdefault(sp, []).append(i)

            if not names or sp != names[-1]:
                names.append(sp)

        if len(names) != len(set(names)):
            raise ValueError("Non contiguous blocks are not supported.")

        species = {}

        for sp, indices in d.items():
            species[sp] = positions[indices, :].copy()

        return cls(lattice, species, names)

    @property
    def species_names(self):
        "Names of species in the same order as found in the raw positions."
        return self._species_names.copy()

    @property
    def atoms(self):
        "List the species names in an order matching `self.raw`."
        atoms = []

        for sp in self._species_names:
            atoms.extend([sp] * len(self.species[sp]))

        return atoms

    @property
    def raw(self):
        """Return an array of atomic positions.

        This can be modified overwritten, but not modified in place.
        """
        return np.vstack([self.species[n] for n in self._species_names])

    @raw.setter
    def raw(self, raw_data):
        offset = 0
        for n in self._species_names:
            slc = slice(offset, offset + len(self.species[n]), 1)
            self.species[n] = raw_data[slc]
            offset += len(self.species[n])

    @property
    def system_name(self):
        """The name of the system, eventually generated from formula.

        Can be overwritten.
        """
        if self._system_name:
            return self._system_name
        else:
            species = list(self.species.items())
            # sort by increasing electronegativity
            species.sort(key=lambda p: electronegativity[p[0]])
            return " ".join(f"{label}{len(pos)}" for label, pos in species)

    @system_name.setter
    def system_name(self, val):
        self._system_name = val if val is None else str(val)

    def copy(self):
        return self.__class__(
            self.lattice.copy(),
            {k: a.copy() for k, a in self.species.items()},
            species_names=self._species_names,
        )

    def sp_at(self, i):
        j = i

        for sp in self._species_names:
            if i < len(self.species[sp]):
                return sp
            else:
                i -= len(self.species[sp])

        raise ValueError(f"{j} is not a valid atom index.")

    def get_offset(self, sp):
        if sp not in self.species:
            raise ValueError(f"{sp} is not present in this structure.")

        offset = 0
        for sp_b in self._species_names:
            if sp_b == sp:
                return offset
            else:
                offset += len(self.species[sp_b])

        raise Exception("Unreachable")
