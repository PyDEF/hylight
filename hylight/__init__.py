"""The Hylight package.

Copyright (c) 2024, Théo Cavignac <theo.cavignac+dev@gmail.com>, The PyDEF team <camille.latouche@cnrs-imn.fr>
Licensed under the EUPL
"""

__version__ = "1.0.0"

from . import constants  # noqa: F401
from . import loader  # noqa: F401
from . import mode  # noqa: F401
from . import mono_mode  # noqa: F401
from . import multi_modes  # noqa: F401

import logging


def setup_logging():
    "Setup module logging."

    class H(logging.Handler):
        def emit(self, record):
            print(self.format(record))

    h = H()

    logging.getLogger("hylight").addHandler(h)


setup_logging()
