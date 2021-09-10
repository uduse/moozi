# %%
import functools
import pickle
import random
import typing
from typing import NamedTuple, Optional

import acme
import acme.jax.utils
import acme.wrappers
import anytree
import chex
import dm_env
import jax
import jax.numpy as jnp
import moozi as mz
import numpy as np
import open_spiel
import optax
import reverb
import rlax
import tree
from absl.testing import absltest, parameterized
from acme import datasets as acme_datasets
from acme import specs as acme_specs
from acme.adders.reverb import DEFAULT_PRIORITY_TABLE, EpisodeAdder, episode
from acme.adders.reverb import test_utils as acme_test_utils
from acme.adders.reverb.base import ReverbAdder, Trajectory
from acme.agents import agent as acme_agent
from acme.agents import replay as acme_replay
from acme.environment_loops.open_spiel_environment_loop import OpenSpielEnvironmentLoop
from acme.jax.variable_utils import VariableClient
from acme.utils import tree_utils
from acme.utils.loggers.base import NoOpLogger
from moozi.nn import NeuralNetwork, NeuralNetworkOutput
from moozi.policies.policy import PolicyFeed, PolicyResult
from moozi.replay import Trajectory, make_replay
from nptyping import NDArray
from reverb import rate_limiters
from reverb.trajectory_writer import TrajectoryColumn

# %%
use_jit = True
if use_jit:
    jax.config.update("jax_disable_jit", not use_jit)

platform = "gpu"
jax.config.update("jax_platform_name", platform)

print(jax.devices())

# %%
# env_columns, env_rows = 3, 3
# env_columns, env_rows = 5, 5
env_columns, env_rows = 6, 6
raw_env = open_spiel.python.rl_environment.Environment(
    f"catch(columns={env_columns},rows={env_rows})"
)
env = acme.wrappers.open_spiel_wrapper.OpenSpielWrapper(raw_env)
env = acme.wrappers.SinglePrecisionWrapper(env)
env_spec = acme.specs.make_environment_spec(env)

# %%
seed = 0
master_key = jax.random.PRNGKey(seed)

max_replay_size = 100000
max_episode_length = env.environment.environment.game.max_game_length()
num_unroll_steps = 3
num_stacked_frames = 1
num_td_steps = 100
batch_size = 256
discount = 0.99


# %%
dim_repr = 64
dim_action = env_spec.actions.num_values
frame_shape = env_spec.observations.observation.shape
stacked_frame_shape = (num_stacked_frames,) + frame_shape
nn_spec = mz.nn.NeuralNetworkSpec(
    stacked_frames_shape=stacked_frame_shape,
    dim_repr=dim_repr,
    dim_action=dim_action,
    repr_net_sizes=(128, 128),
    pred_net_sizes=(128, 128),
    dyna_net_sizes=(128, 128),
)
network = mz.nn.get_network(nn_spec)
lr = 2e-3
optimizer = optax.adam(lr)
print(nn_spec)


def convert_timestep(timestep):
    return timestep._replace(observation=timestep.observation[0])


def frame_to_str_gen(frame):
    for irow, row in enumerate(frame):
        for val in row:
            if np.isclose(val, 0.0):
                yield "."
                continue
            assert np.isclose(val, 1), val
            if irow == len(frame) - 1:
                yield "X"
            else:
                yield "O"
        yield "\n"


def frame_to_str(frame):
    return "".join(frame_to_str_gen(frame))
