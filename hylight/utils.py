import numpy as np


def make_cell(val):

    nothing = object()
    var = val

    def cell(val=nothing):
        global var
        if val is not nothing:
            var = val
        return var
    return cell


class InputError(ValueError):
    pass


def gen_translat(lattice: np.ndarray):
    """Generate all translations to adjacent cells

    :param lattice: np.ndarray([a, b, c]) first lattice parameter
    """
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            for k in (-1, 0, 1):
                yield np.array([i, j, k]).dot(lattice)
