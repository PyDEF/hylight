from zipfile import ZIP_DEFLATED, ZipFile
import io

import numpy as np

from .. import __version__
from ..constants import atomic_mass
from .data import manifest, state


def write_xyz(f, mode):
    """Write the coordinates and displacements of a mode in the JMol xyz format.

    :param f: a file-like object to write to.
    :param mode: the mode to dump.
    """
    arr = np.hstack([mode.ref, mode.delta / np.sqrt(atomic_mass)])

    print(len(arr), file=f)
    print(f"Mode {mode.n}", file=f)
    for sp, row in zip(mode.atoms, arr):
        print(f"{sp:2}", *(f"  {x:-12.05e}" for x in row), file=f)


def write_jmol_options(f, opts):
    """Write options in the form of a JMol script.

    :param f: a file like object.
    :param opts: a set of options (see export).
    """
    if "unitcell" in opts:
        print(
            "unitcell [ {0 0 0}",
            format_v(opts["unitcell"][0]),
            format_v(opts["unitcell"][1]),
            format_v(opts["unitcell"][2]),
            "]",
            file=f,
        )
    if "bonds" in opts:
        for sp1, sp2, dmin, dmax in opts["bonds"]:
            print(f"connect {float(dmin):0.02} {float(dmax):0.02} (_{sp1}) (_{sp2})", file=f)

    if "atom_colors" in opts:
        for sp, color in opts["atom_colors"]:
            if color.startswith("#"):
                c = f"x{color[1:7]:>06}"
                print(f"color (_{sp}) [{c}]", file=f)
            else:
                print(f"color (_{sp}) {color}", file=f)


def format_v(v):
    x, y, z = v
    return f"{{ {x:0.5f} {y:0.5f} {z:0.5f} }}"


def export(dest, mode, compression=ZIP_DEFLATED, **opts):
    """Export a mode to JMol zip format.

    :param dest: path to the JMol zip file.
    :param mode: the mode to export.
    :param compression: (optional) zipfile compression algorithm.
    :keyword unitcell: lattice vectors as a 3x3 matrix where vectors are in rows.
    :keyword bonds: a list of (sp_a, sp_b, min_dist, max_dis) where species
    names are strings of names of species and *_dist are interatomic distances
    in Angstrom.
    :keyword atom_colors: a list of (sp, color) where sp is the name of a
    species and color is the name of a color or an HTML hex code (example
    "#FF0000" for pure red).
    """
    with ZipFile(dest, mode="w", compression=compression) as ar:
        ar.writestr("JmolManifest.txt", manifest.encode("utf8"))
        ar.writestr("state.spt", state.encode("utf8"))

        with io.StringIO() as f:
            write_xyz(f, mode)
            ar.writestr("system.xyz", f.getvalue().encode("utf8"))

        with io.StringIO() as f:
            print("// System configuration", file=f)
            print(f"// Generated with Hylight {__version__}", file=f)
            write_jmol_options(f, opts)
            ar.writestr("system.spt", f.getvalue().encode("utf8"))
