from typing import List, Sequence

import acme
from dm_env import Environment
import jax
import jax.numpy as jnp
import numpy as np
import moozi as mz
import open_spiel
import pytest
from acme import types
from acme.core import VariableSource
from acme.jax.variable_utils import VariableClient

from pytest_trio.enable_trio_mode import *

from moozi.core import PolicyFeed


def update_jax_config():
    jax.config.update("jax_disable_jit", True)
    print("conftest JAX: disabled jit")
    jax.config.update("jax_platform_name", "cpu")
    print("conftest JAX: platform cpu")


update_jax_config()


@pytest.fixture
def env() -> Environment:
    raw_env = open_spiel.python.rl_environment.Environment("catch(columns=5,rows=5)")
    env = acme.wrappers.open_spiel_wrapper.OpenSpielWrapper(raw_env)
    env = acme.wrappers.SinglePrecisionWrapper(env)
    return env


@pytest.fixture
def env_spec(env):
    env_spec = acme.specs.make_environment_spec(env)
    return env_spec


@pytest.fixture
def num_stacked_frames():
    return 2


@pytest.fixture
def num_unroll_steps() -> int:
    return 2


@pytest.fixture
def network(env_spec, num_stacked_frames):
    dim_action = env_spec.actions.num_values
    frame_shape = env_spec.observations.observation.shape
    stacked_frames_shape = (num_stacked_frames,) + frame_shape
    nn_spec = mz.nn.NNSpec(
        architecture=mz.nn.ResNetArchitecture,
        stacked_frames_shape=stacked_frames_shape,
        dim_repr=2,
        dim_action=dim_action,
        extra={
            "repr_net_num_blocks": 1,
            "pred_trunk_num_blocks": 1,
            "dyna_trunk_num_blocks": 1,
            "dyna_hidden_num_blocks": 1,
        },
    )
    return mz.nn.build_network(nn_spec)


@pytest.fixture
def random_key():
    return jax.random.PRNGKey(0)


@pytest.fixture
def params(network: mz.nn.NeuralNetwork, random_key):
    return network.init_network(random_key)


@pytest.fixture
def policy_feed(env, env_spec, num_frames, random_key) -> PolicyFeed:
    legal_actions_mask = np.ones(env_spec.actions.num_values)
    # legal_actions_indices = [1, 2, 3]
    # legal_actions_mask[legal_actions_indices] = 1
    # legal_actions_mask = jnp.array(legal_actions_mask)
    timestep = env.reset()
    frame = timestep.observation[0].observation
    stacked_frames = jnp.stack([frame.copy() for _ in range(num_frames)])

    return PolicyFeed(
        stacked_frames=stacked_frames,
        to_play=0,
        legal_actions_mask=legal_actions_mask,
        random_key=random_key,
    )


@pytest.fixture
def init_ray():
    import ray

    ray.init(ignore_reinit_error=True)
