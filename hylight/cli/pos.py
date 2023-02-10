"""A set of useful tools to manipulate VASP's POSCAR files.

License:
    Copyright (C) 2023  PyDEF development team

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
from .script_utils import (
    MultiCmd,
    positional,
    flag,
    optional,
    error_catch,
    error,
)

from ..vasp.common import Poscar
from ..utils import periodic_diff
from ..constants import masses
import numpy as np

cmd = MultiCmd(description=__doc__)


@cmd.subcmd(
    positional("POSCAR_REF", type=str, help="Path to the reference POSCAR path."),
    positional("POSCAR_DISP", type=str, help="Path to the displaced POSCAR path."),
    flag("--mass-weighted", "-m", help="Compute nuclear mass weighted distance."),
)
def dist(opts):
    "Compute the total norm of the distorsion between two positions of a system."
    poscar_a = Poscar.from_file(opts.poscar_ref)
    poscar_b = Poscar.from_file(opts.poscar_disp)

    diff = get_diff(poscar_a, poscar_b)

    if opts.mass_weighted:
        m = np.array([masses[sp] for sp in poscar_a.atoms]).reshape((-1, 1))
        print(
            "Distance (A):",
            np.sqrt(np.sum(diff**2 * m) / np.sum(m)),
        )
    else:
        print("Distance (A):", np.linalg.norm(diff))


@cmd.subcmd(
    positional("POSCAR_REF", type=str, help="Path to the reference POSCAR path."),
    positional("POSCAR_DISP", type=str, help="Path to the displaced POSCAR path."),
    flag("--direct", "-d", help="Output in lattice coordinates."),
    flag(
        "--jmol",
        "-J",
        help="Create a JMol file from the distorsion (if --dest is not used, name it after the POSCAR_REF).",
    ),
    optional("--dest", "-o", help="Path to the output file."),
    optional(
        "--bond",
        "-b",
        action="append",
        help="(JMol) specification of a bond (ex: Ti,O,2.1 for a Ti-O bond up to 2.1 A)",
    ),
    optional(
        "--color",
        "-c",
        action="append",
        help="(JMol) specify an atom color (ex: Al,#000090)",
    ),
)
def diff(opts):
    """Compute the displacement between two structures."""
    poscar_a = Poscar.from_file(opts.poscar_ref)
    poscar_b = Poscar.from_file(opts.poscar_disp)

    diff = get_diff(poscar_a, poscar_b)

    norms = np.linalg.norm(diff, axis=-1)

    if opts.jmol:
        from ..jmol import export_disp

        if opts.dest:
            dest = opts.dest
        else:
            dest = opts.poscar_ref + ".jmol"

        jmol_opts = {}

        if opts.bond:
            jmol_opts["bonds"] = []
            with error_catch():
                for s in opts.bond:
                    at1, at2, *lengths = s.split(",")
                    if len(lengths) == 1:
                        jmol_opts["bonds"].append((at1, at2, 0., float(lengths[0])))
                    elif len(lengths) == 2:
                        jmol_opts["bonds"].append((at1, at2, float(lengths[0]), float(lengths[1])))
                    else:
                        raise ValueError("For bond {at1}-{at2}: you need to at least specify the max distance (ex: {at1},{at2},2.2).")

        if opts.color:
            jmol_opts["colors"] = []
            with error_catch():
                for s in opts.color:
                    at, color = s.split(",")

                    if not is_hex_code(color):
                        raise ValueError(f"{color} is not a valid color code.")

                    jmol_opts["colors"].append((at, color))

        with error_catch():
            export_disp(dest, poscar_a, diff, **jmol_opts)

        return

    if opts.direct:
        diff = diff @ np.linalg.inv(poscar_a.cell_parameters)

    for at, (x, y, z), n in zip(
        (spec(poscar_a, i) for i in range(len(poscar_a.atoms))), diff, norms
    ):
        print(f"{at:<4}:", f"{x:9.05f}, {y:9.05f}, {z:9.05f} ({n:9.05f} A)")


def get_diff(poscar_a, poscar_b):
    if poscar_a.system_name != poscar_b.system_name:
        error("This tool can only compare positions from the same system.")

    elif np.linalg.norm(poscar_a.cell_parameters - poscar_b.cell_parameters) > 1.0e-7:
        error("This tool can only compare positions in the same cell size.")

    return periodic_diff(poscar_a.cell_parameters, poscar_a.raw, poscar_b.raw)


def spec(poscar, sp):
    i = sp
    for n in poscar._species_names:
        if i < len(poscar.species[n]):
            return f"{n}{i + 1}"
        else:
            i -= len(poscar.species[n])

    raise ValueError(f"sp is out of bounds: i >= {len(poscar.raw)}")


def is_hex_code(code):
    "Predicate to identify HTML like hexadecimal color codes."
    return (
        len(code) in {7, 4}
        and code[0] == "#"
        and not set(code[1:]).difference("ABCDEFabcdef0123456789")
    )
