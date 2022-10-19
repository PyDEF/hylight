import os.path as op
import gzip

from .script_utils import MultiCmd, positional, optional, error_catch
from ..pkl import archive_modes


pkl = MultiCmd(
    description="""\
Read a file containing vibrational modes and store them into a fast-read file.
"""
)


@pkl.subcmd(
    positional("SOURCE", help="path to the OUTCAR."),
    positional("DEST", help="path of the destionation file.", default=None),
)
def vasp(opts):
    """Convert a VASP OUTCAR to a Hylight archive."""
    if opts.dest is None:
        dest = opts.source + ".npz"
    else:
        dest = opts.dest

    from ..vasp.loader import load_phonons

    with error_catch():
        modes = load_phonons(opts.source)
        archive_modes(modes, dest)

    summary(modes, dest)

    return 0


@pkl.subcmd(
    positional("SOURCE", help="path to the Hylight archive."),
    positional("DEST", help="path of the destionation file."),
)
def hy(opts):
    """Convert a Hylight archive to another form of Hylight archive.

    Use this to convert from pickle to hdf5 or reverse.
    The expected format of the input, and the target format are determined from
    file extensions.
    .h5 and .hdf5 implies the use of HDF5, everything else implies pickle.
    You can add a second extension .gz after the first one, to use GZip
    compression.
    """
    from ..vasp.loader import load_phonons

    with error_catch():
        modes = load_phonons(opts.source)
        archive_modes(modes, opts.dest)

    summary(modes, opts.dest)

    return 0


@pkl.subcmd(
    positional("SOURCE", help="path to a phonopy output file."),
    positional("DEST", help="path of the destionation file.", default=None),
    positional("PHONOPY", help="path to the phonopy output file.", default=None),
)
def phonopy(opts):
    """Convert a Phonopy output file into a Hylight archive.

    The Phonopy file can be one of qpoints.hdf5, qpoints.hdf5.gz, band.hdf5,
    band.hdf5.gz, qpoints.yaml, band.yaml.
    If the file is *not* band.yaml, there should be a phonopy.yaml file in the
    same directory.
    """
    from ..phonopy.loader import (
        load_phonons_bandsh5,
        load_phonons_bandyaml,
        load_phonons_qpointsh5,
        load_phonons_qpointsyaml,
    )

    if opts.dest is None:
        dest = opts.source + ".npz"
    else:
        dest = opts.dest

    source = opts.source

    if op.basename(source) != "band.yaml":
        if opts.phonopy_yaml is None:
            phyaml = op.join(op.dirname(source), "phonopy.yaml")
        else:
            phyaml = opts.phonopy_yaml

    def load_phonons(_):
        if op.basename(source) == "qpoints.hdf5":
            return load_phonons_qpointsh5(source, phyaml)

        elif op.basename(source) == "qpoints.hdf5.gz":
            return load_phonons_qpointsh5(source, phyaml, op=gzip.open)

        elif op.basename(source) == "band.hdf5":
            return load_phonons_bandsh5(source, phyaml)

        elif op.basename(source) == "band.hdf5.gz":
            return load_phonons_bandsh5(source, phyaml, op=gzip.open)

        elif op.basename(source) == "qpoints.yaml":
            return load_phonons_qpointsyaml(source, phyaml)

        elif op.basename(source) == "band.yaml":
            return load_phonons_bandyaml(source)

        else:
            raise FileNotFoundError("No known file to extract modes from.")

    with error_catch():
        modes = load_phonons(opts.source)
        archive_modes(modes, dest)

    summary(modes, dest)

    return 0


@pkl.subcmd(
    positional("SOURCE", help="path of the log file."),
    positional("DEST", help="path of the destionation file.", default=None),
)
def crystal(opts):
    """Convert a CRYSTAL log file into a Hylight archive."""
    if opts.dest is None:
        dest = opts.source + ".npz"
    else:
        dest = opts.dest

    from ..crystal.loader import load_phonons

    with error_catch():
        modes = load_phonons(opts.source)
        archive_modes(modes, dest)

    summary(modes, dest)

    return 0


def summary(data, dest):
    modes, _, _ = data
    print(f"Wrote {len(modes)} modes in {dest}")
