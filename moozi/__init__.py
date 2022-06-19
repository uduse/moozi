from jax.config import config as jax_config

jax_config.update("jax_debug_nans", True)

from .core import *
from . import types
from . import utils
from . import nn
from . import replay
from . import jaxboard
from . import logging
