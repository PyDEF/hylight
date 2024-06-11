"""Pervasive utilities.
"""

# Copyright (c) 2024, Théo Cavignac <theo.cavignac+dev@gmail.com>, The PyDEF team <camille.latouche@cnrs-imn.fr>
# Licensed under the EUPL
import re

import numpy as np
from scipy.interpolate import interp1d


class InputError(ValueError):
    "An exception raised when the input files are not as expected."


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
    dp = ref - disp

    dfrac = np.remainder(dp @ np.linalg.inv(lattice) + 0.5, 1.0) - 0.5

    return dfrac @ lattice


def periodic_dist(lattice, ref, disp):
    "Compute the distance between ref and disp, accounting for periodic conditions."
    dp = ref - disp

    dfrac = np.remainder(dp @ np.linalg.inv(lattice) + 0.5, 1.0) - 0.5

    return np.linalg.norm(dfrac @ lattice, axis=-1)


def gaussian(e, sigma, standard=True):
    """Evaluate a Gaussian function on e.

    :param e: abscissa
    :param sigma: standard deviation
    :param standard:

        - if True the curve is normalized to have an area of 1
        - if False the curve is normalized to have a maximum of 1
    """
    if standard:
        return np.exp(-(e**2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
    else:
        return np.exp(-(e**2) / (2 * sigma**2))


def parse_formatted_table(lines, format):
    """Parse a table of numbers from a list of string with a regex.

    :param lines: a list of string representing the lines of the table
    :param format: a re.Pattern or a str representing the format of the line
      It should fullmatch each line or a ValueError exception will be raised.
      All the groups defined in format will be converted to float64 by numpy.
    :returns: a np.array of dimension (len(lines), {number_of_groups})

    Example:
    --------
        >>> parse_formatted_table(["a=0.56 b=0.8 c=0.9"], "a=(.*) b=(.*) c=(.*)")
        np.array([[0.56, 0.8, 0.9]])

    """
    if isinstance(format, re.Pattern):
        matcher = format
    else:
        matcher = re.compile(format)

    res = []
    for line in lines:
        m = matcher.match(line)

        if not m:
            raise ValueError(f"No match for {format!r} with line: {line!r}")

        res.append(m.groups())

    assert len(res) >= 1 and all(len(res[0]) == len(r) for r in res)
    return np.array(res, dtype=float)
