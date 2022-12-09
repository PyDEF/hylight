__version__ = "0.2.0"

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
