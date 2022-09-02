import os.path as op
import gzip

from .script_utils import MultiCmd, positional, optional, error_catch
from ..pkl import pickle_modes


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
    if opts.dest is None:
        dest = opts.source + ".pkl"
    else:
        dest = opts.dest

    from ..vasp.loader import load_phonons

    with error_catch():
        pickle_modes(opts.source, dest, load_phonons)

    return 0


@pkl.subcmd(
    positional("SOURCE", help="path to a phonopy output file."),
    positional("DEST", help="path of the destionation file.", default=None),
    positional("PHONOPY_YAML", help="path to the phonopy.yaml file.", default=None),
)
def phonopy(opts):
    from ..phonopy.loader import (
        load_phonons_bandsh5,
        load_phonons_bandyaml,
        load_phonons_qpointsh5,
        load_phonons_qpointsyaml,
        Struct,
    )

    if opts.dest is None:
        dest = opts.source + ".pkl"
    else:
        dest = opts.dest

    source = opts.source

    if op.basename(source) != "band.yaml":
        if opts.phonopy_yaml is None:
            phyaml = op.join(op.dirname(source), "phonopy.yaml")
        else:
            phyaml = opts.phonopy_yaml

    def wrap(_):
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
        pickle_modes(opts.source, dest, wrap)

    return 0


@pkl.subcmd(
    positional("SOURCE", help="path of the log file."),
    positional("DEST", help="path of the destionation file.", default=None),
)
def crystal(opts):
    if opts.dest is None:
        dest = opts.source + ".pkl"
    else:
        dest = opts.dest

    from ..crystal.loader import load_phonons

    with error_catch():
        pickle_modes(opts.source, dest, load_phonons)

    return 0
