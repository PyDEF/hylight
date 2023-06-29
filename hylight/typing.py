"""Helper module for type annotations.

Should help with variable support of typing across versions of python and libraries.
"""
# License:
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
