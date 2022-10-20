import numpy as np

from .constants import electronegativity


class Struct:
    def __init__(self, cell_parameters, species, species_names=None):
        """Store all informations about a unit cell.

        :param cell_parameters: a 3x3 np.array with lattice vectors in line
        :param species: a dict[str, np.array] where the key is the name of the
          species and the array list positions.
          WARNING: Positions are in cartesian representation, not in fractional
          representation. Unit is Angstrom.
        """
        self.cell_parameters = cell_parameters
        self.species = species
        self._system_name = None
        if species_names is None:
            self._species_names = sorted(
                self.species.keys(), key=lambda p: electronegativity[p]
            )
        else:
            self._species_names = list(species_names)

    @property
    def raw(self):
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
            self.cell_parameters.copy(),
            {
                k: a.copy()
                for k, a in self.species.items()
            },
            species_names=self._species_names,
        )
