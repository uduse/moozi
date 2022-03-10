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


from moozi.core import PolicyFeed, make_catch


def update_jax_config():
    jax.config.update("jax_disable_jit", True)
    print("conftest JAX: disabled jit")
    jax.config.update("jax_platform_name", "cpu")
    print("conftest JAX: platform cpu")


update_jax_config()


@pytest.fixture
def env() -> Environment:
    return make_catch()[0]


@pytest.fixture
def env_spec(env):
    return make_catch()[1]


@pytest.fixture
def num_stacked_frames():
    return 2


@pytest.fixture
def num_unroll_steps() -> int:
    return 2


@pytest.fixture
def model(env_spec, num_stacked_frames):
    dim_action = env_spec.actions.num_values
    frame_shape = env_spec.observations.observation.shape
    stacked_frames_shape = frame_shape[:-1] + (num_stacked_frames * frame_shape[-1],)
    nn_spec = mz.nn.ResNetSpec(
        stacked_frames_shape=stacked_frames_shape,
        dim_repr=2,
        dim_action=dim_action,
        # use default values for other parameters
    )
    return mz.nn.make_model(mz.nn.ResNetArchitecture, nn_spec)


@pytest.fixture
def random_key():
    return jax.random.PRNGKey(0)


@pytest.fixture
def params_and_state(model: mz.nn.NNModel, random_key):
    return model.init_params_and_state(random_key)


@pytest.fixture
def params(params_and_state):
    return params_and_state[0]


@pytest.fixture
def state(params_and_state):
    return params_and_state[1]


@pytest.fixture
def policy_feed(env, env_spec, num_stacked_frames, random_key) -> PolicyFeed:
    legal_actions_mask = np.ones(env_spec.actions.num_values)
    timestep = env.reset()
    frame = timestep.observation[0].observation
    stacked_frames = frame.repeat(num_stacked_frames, axis=-1)

    return PolicyFeed(
        stacked_frames=stacked_frames,
        to_play=0,
        legal_actions_mask=legal_actions_mask,
        random_key=random_key,
    )


# TODO: this fixutre seems redundent
@pytest.fixture
def init_ray():
    import ray

    ray.init(ignore_reinit_error=True)
