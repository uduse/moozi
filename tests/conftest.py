import os
from typing import List, Sequence

import acme
import jax
import jax.numpy as jnp
import moozi as mz
import numpy as np
import open_spiel
import pytest
from acme import types
from acme.core import VariableSource
from acme.jax.variable_utils import VariableClient
from dm_env import Environment
from moozi.core import PolicyFeed
from moozi.core.env import GIIEnv
from moozi.core.scalar_transform import ScalarTransform
from moozi.gii import GII


def update_jax_config():
    jax.config.update("jax_disable_jit", True)
    print("conftest JAX: disabled jit")
    jax.config.update("jax_platform_name", "cpu")
    print("conftest JAX: platform cpu")


# NOTE: uncomment line to make easier to debug JAX
# update_jax_config()


@pytest.fixture(scope="session", params=["OpenSpiel:catch", "MinAtar:Breakout-v1"])
def env(request) -> GIIEnv:
    return GIIEnv.new(request.param)


@pytest.fixture(scope="session")
def history_length():
    return 2


@pytest.fixture(scope="session")
def num_unroll_steps() -> int:
    return 2


@pytest.fixture(scope="session")
def scalar_transform() -> ScalarTransform:
    return ScalarTransform.new(support_min=-2, support_max=2, contract=True)


@pytest.fixture(scope="session")
def model(env: GIIEnv, history_length, scalar_transform):
    dim_action = env.spec.dim_action
    frame_shape = env.spec.frame.shape
    nn_spec = mz.nn.NNSpec(
        dim_action=dim_action,
        num_players=env.spec.num_players,
        history_length=history_length,
        frame_rows=frame_shape[0],
        frame_cols=frame_shape[1],
        frame_channels=frame_shape[2],
        repr_rows=frame_shape[0],
        repr_cols=frame_shape[1],
        repr_channels=frame_shape[2],
        scalar_transform=scalar_transform,
    )
    return mz.nn.make_model(mz.nn.NaiveArchitecture, nn_spec)


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
