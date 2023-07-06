"""Read a file containing vibrational modes and store them into a fast-read file.

License
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
import os
import os.path as op
import gzip
from itertools import cycle

from .script_utils import MultiCmd, positional, optional, error_catch, flag
from ..npz import archive_modes, load_phonons
from ..constants import eV_in_J, cm1_in_J, atomic_mass, hbar_si


cmd = MultiCmd(description=__doc__)


@cmd.subcmd(
    positional("SOURCE", help="path to an input file."),
    positional("DEST", help="path of the destionation file.", default=None),
    positional("PHONOPY_YAML", help="path to the phonopy.yaml file if needed.", default=None),
    optional("--from", "-f", dest="from_", default="auto", help="Input format."),
    optional("--with-lattice", "-l", default=None, help="Reference POSCAR for the lattice."),
    optional("--tol", "-t", default=1e-6, type=float, help="Numerical tolerance to lattice vector mismatch (used with --with-lattice)."),
    flag("--asr", "-A", help="Project the dynmical matrix to enforce accoustic sum rule."),
)
def convert(opts):
    """Convert a set of modes from an upstream software to a Hylight archive.

    Supported formats include "auto", "crystal", "npz", "phonopy" and "vasp".

    In "auto" mode, VASP files are recognized if they end with ".vasp" or are named "OUTCAR".
    CRYSTAL files are recognized if they end with ".log".
    npz files are recognized if they end with ".npz".
    Phonopy files are recognized if they are one of "band.hdf5", "band.hdf5.gz",
    "qpoints.hdf5", "qpoints.hdf5.gz", "qpoints.yaml", "band.yaml".

    In "phonopy" mode, if the file is *not* band.yaml, there should be a
    "phonopy.yaml" file in the same directory.

    """
    import numpy as np

    if opts.dest is None:
        dest = opts.source.lstrip('/') + ".npz"
    else:
        dest = opts.dest

    with error_catch():
        modes = multi_loader(opts.source, opts.from_, opts.phonopy_yaml)

    summary(modes, opts.source)

    if opts.with_lattice:
        from ..vasp.common import Poscar

        p = Poscar.from_file(opts.with_lattice)

        for mode in modes[0]:
            mode.set_lattice(p.lattice, tol=opts.tol)

    if opts.asr:
        from ..mode import dynamical_matrix, project_on_asr, modes_from_dynmat
        phonons, pops, masses = modes
        dmat = dynamical_matrix(phonons)
        m0 = modes[0][0]

        dmat_proj = project_on_asr(dmat, masses)

        d = np.linalg.norm(dmat - dmat_proj)**0.5 * hbar_si / cm1_in_J
        print("Distance to original matrix:", d, "cm1")

        phonons = modes_from_dynmat(m0.lattice, m0.atoms, masses, m0.ref, dmat_proj)
        modes = (phonons, pops, masses)


    with error_catch():
        archive_modes(modes, dest)

    print(f"Wrote {dest}")

    return 0


@cmd.subcmd(
    positional("SOURCE", help="path to a phonopy output file."),
    flag("--cartesian", "-c", help="Use cartesian coordinates."),
)
def show_ref(opts):
    """Write the reference position to STDOUT using the POSCAR format."""

    from sys import stdout
    from ..npz import load_phonons
    from ..vasp.common import Poscar

    modes, _, _ = load_phonons(opts.source)

    m0 = modes[0]
    p = Poscar(m0.lattice, {"H": m0.ref})
    p.to_stream(stdout, cartesian=opts.cartesian)

def multi_loader(source, from_, phonopy_yaml):
    if from_ not in {"vasp", "crystal", "npz", "phonopy", "auto"}:
        raise ValueError(f"{from_} is not a known input format.")

    if from_ == "vasp" or source.endswith(".vasp") or op.basename(source) == "OUTCAR":
        from ..vasp.loader import load_phonons
        return load_phonons(source)

    if from_ == "crystal" or source.endswith(".log"):
        from ..crystal.loader import load_phonons
        return load_phonons(source)

    if from_ == "npz" or source.endswith(".npz"):
        from ..npz import load_phonons
        return load_phonons(source)

    if phonopy_yaml is None:
        phyaml = op.join(op.dirname(source), "phonopy.yaml")
    else:
        phyaml = phonopy_yaml

    if op.basename(source) == "qpoints.hdf5":
        from ..phonopy.loader import load_phonons_qpointsh5
        return load_phonons_qpointsh5(source, phyaml)

    elif op.basename(source) == "qpoints.hdf5.gz":
        from ..phonopy.loader import load_phonons_qpointsh5
        return load_phonons_qpointsh5(source, phyaml, op=gzip.open)

    elif op.basename(source) == "band.hdf5":
        from ..phonopy.loader import load_phonons_bandsh5
        return load_phonons_bandsh5(source, phyaml)

    elif op.basename(source) == "band.hdf5.gz":
        from ..phonopy.loader import load_phonons_bandsh5
        return load_phonons_bandsh5(source, phyaml, op=gzip.open)

    elif op.basename(source) == "qpoints.yaml":
        from ..phonopy.loader import load_phonons_qpointsyaml
        return load_phonons_qpointsyaml(source, phyaml)

    elif op.basename(source) == "band.yaml":
        from ..phonopy.loader import load_phonons_bandyaml
        return load_phonons_bandyaml(source)

    elif from_ == "phonopy":
        raise FileNotFoundError("No known phonopy input found.")

    else:
        raise ValueError("Could not determine the input format. Please use the --format parameter.")


@cmd.subcmd(
    positional("SOURCE", help="path of the npz file."),
    positional("SELECTION", type=int, help="index of the mode to export."),
    optional("--dest", "-o", default=None, help="path of the destionation file."),
    optional(
        "--bond",
        "-b",
        action="append",
        help="specification of a bond (ex: Ti,O,2.1 for a Ti-O bond up to 2.1 A)",
    ),
    optional(
        "--color", "-c", action="append", help="specify an atom color (ex: Al,#000090)"
    ),
    optional("--ref", "-r", default=None, help="A reference POSCAR"),
)
def jmol(opts):
    """Extract a mode from a npz file and produce a JMol file."""
    from ..jmol import export

    with error_catch():
        modes, _, _ = load_phonons(opts.source)
        m = modes[opts.selection - 1]

    mode_summary(m)

    if opts.dest is not None:
        dest = opts.dest
    else:
        dest = os.path.splitext(opts.source)[0] + f"_{opts.selection}.jmol"

    x_opts = parse_opts(opts)

    with error_catch():
        export(dest, m, **x_opts)


def mode_summary(mode):
    print("Index:", mode.n)
    print("Real:", "yes" if mode.real else "no")
    print("Energy (meV):", round(mode.energy * 1000 / eV_in_J, 2))


def parse_opts(opts):
    x_opts = {}

    if opts.bond:
        x_opts["bonds"] = [
            (sp1, sp2, 0.0, float(dmax))
            for sp1, sp2, dmax in (s.split(",") for s in opts.bond)
        ]

    if opts.color:
        x_opts["atom_colors"] = [
            (sp, color) for sp, color in (s.split(",") for s in opts.color)
        ]

    if opts.ref:
        from ..vasp.common import Poscar

        if opts.ref:
            p = Poscar.from_file(opts.ref)

            x_opts["unitcell"] = p.lattice

    return x_opts


@cmd.subcmd(
    positional("SOURCE", help="path to the mode archive."),
)
def show(opts):
    """Produce a set of figures to vizualize the modes."""
    from ..multi_phonons import dynmatshow
    from ..mode import dynamical_matrix
    from matplotlib.pyplot import show

    phonons, _, _ = load_phonons(opts.source)

    blocks = []

    colors = cycle(
        [
            "red",
            "blue",
            "orange",
            "purple",
            "green",
            "pink",
            "yellow",
        ]
    )

    prev = None
    acc = 0
    for at in phonons[0].atoms:
        if not prev:
            prev = at
            acc = 1
        elif at != prev:
            if acc:
                blocks.append(
                    (prev, acc, next(colors)),
                )
                acc = 1
                prev = at
        else:
            acc += 1

    if acc:
        blocks.append(
            (prev, acc, next(colors)),
        )
        acc = 1
        prev = at

    dynmat = dynamical_matrix(phonons)
    fig, ax = dynmatshow(dynmat, blocks=blocks)

    n = len(phonons)
    m = sum(not m.real for m in phonons)
    e_re = max((m.energy for m in phonons if m.real), default=0)
    e_im = max((m.energy for m in phonons if not m.real), default=0)

    print(f"There are {n} modes, among which {m} are unstable.")
    print(
        f"Maximum real frequency is {e_re / eV_in_J * 1e3:0.03f} meV / {e_re / cm1_in_J:0.03f} cm1."
    )
    print(
        f"Maximum imaginary frequency is {e_im / eV_in_J * 1e3:0.03f} meV / {e_im / cm1_in_J:0.03f} cm1."
    )

    show()


def summary(data, src):
    modes, _, _ = data
    print(f"Loaded {len(modes)} modes from {src}.")
