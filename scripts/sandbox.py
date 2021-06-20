# %%
import random
import typing
from typing import NamedTuple, Optional

import acme
import acme.jax.utils
import acme.jax.variable_utils
import acme.wrappers
import chex
import dm_env
import jax
import jax.numpy as jnp
import moozi as mz
import numpy as np
import open_spiel
import optax
import reverb
import tree
from absl.testing import absltest, parameterized
from acme import datasets as acme_datasets
from acme import specs as acme_specs
from acme.adders.reverb import DEFAULT_PRIORITY_TABLE, EpisodeAdder
from acme.adders.reverb import test_utils as acme_test_utils
from acme.adders.reverb.base import ReverbAdder, Trajectory
from acme.agents import agent as acme_agent
from acme.agents import replay as acme_replay
from acme.environment_loops.open_spiel_environment_loop import OpenSpielEnvironmentLoop
from acme.utils.loggers.base import NoOpLogger
from moozi.replay import make_replay
from nptyping import NDArray
from reverb import rate_limiters
from reverb.trajectory_writer import TrajectoryColumn

# %%
use_jit = True
if use_jit:
    jax.config.update("jax_disable_jit", not use_jit)

platform = "cpu"
jax.config.update("jax_platform_name", platform)

# %%
raw_env = open_spiel.python.rl_environment.Environment("catch(columns=7,rows=5)")
env = acme.wrappers.open_spiel_wrapper.OpenSpielWrapper(raw_env)
env = acme.wrappers.SinglePrecisionWrapper(env)
env_spec = acme.specs.make_environment_spec(env)

# %%
seed = 0
key = jax.random.PRNGKey(seed)
max_replay_size = 1000
max_episode_length = 5
num_unroll_steps = 2
num_stacked_frames = 2
num_td_steps = 4
batch_size = 32

# %%
reverb_replay = make_replay(
    env_spec, max_episode_length=max_episode_length, batch_size=batch_size
)

# %%
key, new_key = jax.random.split(key)
random_actor = mz.Actor(
    env_spec=env_spec,
    policy=mz.policies.RandomPolicy(),
    adder=reverb_replay.adder,
    random_key=new_key,
)

# %%
loop = OpenSpielEnvironmentLoop(
    environment=env, actors=[random_actor], logger=NoOpLogger()
)
loop.run_episode()
