"""Hylight

License
    Copyright (C) 2023  PyDEF development team

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
__version__ = "1.0.0"

from . import constants  # noqa: F401
from . import loader  # noqa: F401
from . import mode  # noqa: F401
from . import mono_phonon  # noqa: F401
from . import multi_phonons  # noqa: F401

import logging


def setup_logging():
    class H(logging.Handler):
        def emit(self, record):
            print(self.format(record))

    h = H()

    logging.getLogger("hylight").addHandler(h)


setup_logging()
