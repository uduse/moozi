from jax.config import config as jax_config

jax_config.update("jax_debug_nans", True)

from .config import Config
from . import types
from . import utils
from . import nerual_network, nerual_network as nn  # alias
from . import policies
from . import replay
from . import loss
from . import jaxboard
from . import logging
# from . import actor
from . import learner
from . import agent

# from .action_history import ActionHistory

# from .config import Config
# from .player import Player
# from .node import Node
# from .game import Game

# from . import mcts
# from . import utils
# from . import config
