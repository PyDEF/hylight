import pickle


def pickle_modes(source, dest, load_phonons):
    t = load_phonons(source)
    with open(dest, "wb") as f:
        pickle.dump(("hylight-pkl-modes", t), f)


def load_phonons(source):
    with open(source, "rb") as f:
        try:
            label, data = pickle.load(f)
        except ValueError:
            raise ValueError("This is not a pickled mode file.")

    if label != "hylight-pkl-modes":
        raise ValueError("This is not a pickled mode file.")

    return data
