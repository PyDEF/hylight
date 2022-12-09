import numpy as np
from scipy.interpolate import interp1d


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


def measure_fwhm(x, y):
    """Measure the full width at half maximum of a given spectra.

    Warning: It may fail if there are more than one band that reach half
    maximum in the array. In this case you may want to use select_interval to
    make a window around a single band.
    :param x: the energy array
    :param y: the intensity array
    :return: FWHM in the same unit as x.
    """
    mx = np.max(y)

    x_ = x[y > (mx / 2)]
    return np.max(x_) - np.min(x_)


def select_interval(x, y, emin, emax, normalize=False, npoints=None):
    """Extract an interval of a spectra and return the windows x and y arrays.

    :param x: x array
    :param y: y array
    :param emin: lower bound for the window
    :param emax: higher bound for the window
    :param normalize: (optional, False) if true, the result y array is normalized
    :param npoints: (optional, None) if an integer, the result arrays will be
    interpolated to contains exactly npoints linearly distributed between emin
    and emax.
    :return: (windowed_x, windowed_y)
    """
    slice_ = (x > emin) * (x < emax)
    xs, ys = x[slice_], y[slice_] / (np.max(y[slice_]) if normalize else 1.0)

    if npoints is not None:
        emin = max(np.min(xs), emin)
        emax = min(np.max(xs), emax)
        xint = np.linspace(emin, emax, npoints)
        return xint, interp1d(xs, ys)(xint)

    return xs, ys


def periodic_diff(lattice, ref, disp):
    "Compute the displacement between ref and disp, accounting for periodic conditions."
    dp = (disp - ref).reshape((1, -1, 3))
    t = np.array(list(gen_translat(lattice))).reshape((27, 1, 3))
    d = dp - t

    # norms is an array of length of delta
    norms = np.linalg.norm(d, axis=2)  # shape = (27, n)
    best_translat = np.argmin(norms, axis=0)  # shape = (n,)

    n = d.shape[1]
    return d[best_translat, list(range(n)), :]


def gaussian(e, sigma, standard=True):
    if standard:
        return np.exp(-(e**2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
    else:
        return np.exp(-(e**2) / (2 * sigma**2))
