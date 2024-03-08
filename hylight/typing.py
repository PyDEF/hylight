"""Helper module for type annotations.

Should help with variable support of typing across versions of python and libraries.
"""
# Copyright (c) 2024, Théo Cavignac <theo.cavignac+dev@gmail.com>, The PyDEF team <camille.latouche@cnrs-imn.fr>
# Licensed under the EUPL
from __future__ import annotations
from typing import Type
import numpy

# # Unfortunatly, numpy.typing is a feature of Numpy 1.20
# # and mypy does not support dynamic type aliases
# from numpy.typing import NDArray
# from numpy import float64, bool_
#
# FArray = NDArray[float64]
# BArray = NDArray[bool_]
FArray = numpy.ndarray
BArray = numpy.ndarray
