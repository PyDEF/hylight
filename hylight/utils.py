"Pervasive utilities."
# License
#     Copyright (C) 2023  PyDEF development team
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.
import numpy as np
from scipy.interpolate import interp1d


def make_cell(val):
    """An helper function to create a mutable cell/box.

    :param val: initial value
    :returns: a function c

    Example:
        >>> c = make_cell(42)
        >>> c()
        42
        >>> c(23)
        >>> c()
        23
    """
    nothing = object()
    var = val

    def cell(val=nothing):
        global var
        if val is not nothing:
            var = val
        return var

    return cell


class InputError(ValueError):
    "An exception raised when the input files are not as expected."

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
    """Measure the full width at half maximum of a given spectrum.

    .. warning::

        It may fail if there are more than one band that reach half
        maximum in the array. In this case you may want to use
        select_interval to make a window around a single band.

    :param x: the energy array
    :param y: the intensity array
    :returns: FWHM in the same unit as x.
    """
    mx = np.max(y)

    x_ = x[y > (mx / 2)]
    return np.max(x_) - np.min(x_)


def select_interval(x, y, emin, emax, normalize=False, npoints=None):
    """Extract an interval of a spectrum and return the windows x and y arrays.

    :param x: x array
    :param y: y array
    :param emin: lower bound for the window
    :param emax: higher bound for the window
    :param normalize: if true, the result y array is normalized
    :param npoints: if an integer, the result arrays will be
        interpolated to contains exactly npoints linearly distributed
        between emin and emax.
    :return: :code:`(windowed_x, windowed_y)`
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
    """A Gaussian function.

    :param e: mean
    :param sigma: standard deviation
    :param standard:

        - if True the curve is normalized to have an area of 1
        - if False the curve is normalized to have a maximum of 1
    """
    if standard:
        return np.exp(-(e**2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
    else:
        return np.exp(-(e**2) / (2 * sigma**2))
