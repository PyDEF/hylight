import numpy as np

from ..utils import InputError
from ..constants import electronegativity


class Poscar:
    def __init__(self, cell_parameters, species, species_names=None):
        """Create a Poscar type object, storing unit cell infos.

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

    @classmethod
    def from_cell(cls, cell):
        species = {}
        positions = cell.positions
        accum = 0
        for name, number in zip(cell.atoms_types, cell.nb_atoms):
            species[name] = positions[accum : accum + number]
            accum += number

        species_names = list(cell.atoms_types)
        params = cell.cell_parameters

        return Poscar(params, species, species_names=species_names)

    @classmethod
    def from_file(cls, filename):
        with open(filename) as f:
            next(f)  # system name
            fac = float(next(f))
            params = fac * np.array(
                [
                    np.array(l.strip().split(), dtype="float")
                    for _, l in zip(range(3), f)
                ]
            )

            labels = next(f).strip().split()
            atoms_pop = list(map(int, next(f).strip().split()))
            if len(labels) != len(atoms_pop):
                raise InputError(f"{filename} is not a coherent POSCAR file.")

            mode = next(f).strip()[0].lower()

            species = {}
            if mode == "d":
                for spec, n in zip(labels, atoms_pop):
                    pos = []
                    for _, line in zip(range(n), f):
                        ls = line.strip()
                        if not ls:
                            raise InputError(
                                f"{filename} is not a coherent POSCAR file."
                            )
                        x, y, z, *_ = ls.split()
                        pos.append(np.array([x, y, z], dtype="float").dot(params))
                    species[spec] = np.array(pos)
            else:
                for spec, n in zip(labels, atoms_pop):
                    pos = []
                    for _, line in zip(range(n), f):
                        ls = line.strip()
                        if not ls:
                            raise InputError(
                                f"{filename} is not a coherent POSCAR file."
                            )
                        x, y, z, *_ = ls.split()
                        pos.append(np.array([x, y, z], dtype="float"))
                    species[spec] = np.array(pos)

            return Poscar(params, species, species_names=labels)

    def to_file(self, path="POSCAR", cartesian=True):
        """Write a POSCAR file

        The property system_name may be set to change the comment at the top of
        the file.
        :param path: path to the file to write
        :param cartesian: if True, write the file in cartesian representation,
          if False, write in fractional representation
        """
        with open(path, "w+") as out:
            species = [
                (n, self.species[n])
                for n in self._species_names
            ]

            out.write(f"{self.system_name}\n")
            out.write("1.0\n")
            np.savetxt(
                out, self.cell_parameters, "%15.12f", delimiter="\t", newline="\n"
            )

            out.write(" ".join(f"{name:6}" for name, _lst in species))
            out.write("\n")
            out.write(" ".join(f"{len(lst):6}" for _name, lst in species))
            out.write("\n")

            if cartesian:
                out.write("Cartesian\n")
                for _name, lst in species:
                    for pos in lst:
                        out.write("  ".join(f"{x:.8f}" for x in pos))
                        out.write("\n")
            else:
                out.write("Direct\n")
                inv_params = np.linalg.inv(self.cell_parameters)
                for _name, lst in species:
                    for pos in lst:
                        d_pos = pos.dot(inv_params)
                        out.write("  ".join(f"{x:.8f}" for x in d_pos))
                        out.write("\n")

