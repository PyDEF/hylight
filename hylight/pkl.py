import pickle


def pickle_modes(source, dest, load_phonons):
    """Load modes from source using load_phonons and store them in dest.

    :param source: the path to a file containing the modes.
    :param dest: the path to write the modes to.
    :return: the data returned by load_phonons.
    """
    t = load_phonons(source)
    with open(dest, "wb") as f:
        pickle.dump(("hylight-pkl-modes", t), f)

    return t


def load_phonons(source):
    """Load modes from a pickled file.
    """
    with open(source, "rb") as f:
        try:
            label, data = pickle.load(f)
        except ValueError:
            raise ValueError("This is not a pickled mode file.")

    if label != "hylight-pkl-modes":
        raise ValueError("This is not a pickled mode file.")

    return data
