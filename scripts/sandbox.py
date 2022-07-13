# %%
import contextlib
import operator
import sys
import uuid
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Set, Tuple, Union

import chex
import gym
import jax
import jax.numpy as jnp
import mctx
import moozi as mz
import numpy as np
import pygraphviz
import ray
import seaborn as sns
import tree
from acme.jax.utils import add_batch_dim
from acme.utils.tree_utils import stack_sequence_fields
from dotenv import load_dotenv
from IPython.display import display
from loguru import logger
from moozi.core import scalar_transform
from moozi.core.env import make_env
from moozi.core.link import link
from moozi.core.scalar_transform import ScalarTransform, make_scalar_transform
from moozi.core.tape import exclude, include, make_tape
from moozi.laws import *
from moozi.laws import MinAtarVisualizer
from moozi.logging import JAXBoardLoggerRemote, TerminalLoggerRemote
from moozi.nn.nn import NNModel, make_model
from moozi.nn.training import (
    _make_obs_from_train_target,
    make_target_from_traj,
    make_training_suite,
)
from moozi.parameter_optimizer import ParameterOptimizer, ParameterServer
from moozi.planner import make_planner
from moozi.replay import ReplayBuffer

# from moozi.rollout_worker import RolloutWorkerWithWeights, make_rollout_workers
from moozi.utils import WallTimer
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont, ImageOps

# %%
import haiku as hk
import jax
import numpy as np
import jax.numpy as jnp
key = jax.random.PRNGKey(1)

class MyModule(hk.Module):
    def __init__(self):
        super().__init__()

    def f(self, x, is_training):
        x = hk.Linear(2)(x)
        x = hk.BatchNorm(True, True, 0.9)(x, is_training)
        x = jax.nn.relu(x)
        return x
        
    def g(self, x, is_training):
        x = hk.Linear(2)(x)
        x = hk.BatchNorm(True, True, 0.9)(x, is_training)
        x = jax.nn.relu(x)
        return x


def multi_transform_target():
    module = MyModule()

    def module_walk(inputs):
        x = module.f(inputs, is_training=True)
        x = module.f(inputs, is_training=False)
        y = module.g(x, is_training=True)
        y = module.g(x, is_training=False)
        return (x, y)

    return module_walk, (module.f, module.g)


hk_transformed = hk.multi_transform_with_state(multi_transform_target)
params, state = hk_transformed.init(key, jnp.ones(2))

x = np.random.randn(2)
print(x)

y1, _ = hk_transformed.apply[0](params, state, key, x, True)
print(y1)

y2, _ = hk_transformed.apply[0](params, state, key, x, False)
print(y2)

# %% 
t = hk.transform_with_state(lambda x, is_training: MyModule().f(x, is_training))
# %%
params, state = t.init(key, x, True)

# %%
x = np.random.randn(2)
print(x)

y1, state1 = t.apply(params, state, key, x, True)
print(y1)

y2, _ = t.apply(params, state, key, x, False)
print(y2)

# %%
