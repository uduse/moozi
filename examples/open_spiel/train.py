# %%
import numpy as np
import jax.numpy as jnp
from dotenv import load_dotenv
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Type

import cloudpickle
import haiku as hk
import jax
import moozi as mz
import ray
from loguru import logger
from moozi.core import HistoryStacker, TrajectoryCollector, make_scalar_transform
from moozi.core.env import GIIEnv, GIIEnvFeed, GIIEnvOut, GIIVecEnv
from moozi.core.types import TrainingState
from moozi.core.vis import BreakthroughVisualizer, visualize_search_tree
from moozi.nn.nn import NNArchitecture, NNSpec
from moozi.parameter_optimizer import ParameterServer
from moozi.planner import Planner
from moozi.replay import ReplayBuffer, ShardedReplayBuffer
from moozi.tournament import Player, Tournament
from moozi.training_worker import TrainingWorker
from omegaconf import DictConfig
from moozi.driver import Driver, get_config

# %%
load_dotenv()
config = get_config()
driver = Driver.setup(config)
driver.start()
driver.run()
