__version__ = "0.2.0"

from . import constants
from . import loader
from . import mode
from . import mono_phonon
from . import multi_phonons

import logging


def setup_logging():
    class H(logging.Handler):
        def emit(self, record):
            print(self.format(record))

    h = H()

    logging.getLogger("hylight").addHandler(h)


setup_logging()
