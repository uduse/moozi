from typing import List, Sequence

import acme
from dm_env import Environment
import jax
import moozi as mz
import open_spiel
import pytest
from acme import types
from acme.core import VariableSource
from acme.jax.variable_utils import VariableClient

from pytest_trio.enable_trio_mode import *


def update_jax_config():
    jax.config.update("jax_disable_jit", True)
    jax.config.update("jax_platform_name", "cpu")


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
def num_frames():
    return 2


@pytest.fixture
def num_unroll_steps() -> int:
    return 2


@pytest.fixture
def network(env_spec, num_stacked_frames):
    dim_action = env_spec.actions.num_values
    frame_shape = env_spec.observations.observation.shape
    stacked_frames_shape = (num_stacked_frames,) + frame_shape
    nn_spec = mz.nn.NeuralNetworkSpec(
        stacked_frames_shape=stacked_frames_shape,
        dim_repr=2,
        dim_action=dim_action,
        repr_net_sizes=(2, 2),
        pred_net_sizes=(2, 2),
        dyna_net_sizes=(2, 2),
    )
    return mz.nn.get_network(nn_spec)


@pytest.fixture
def random_key():
    return jax.random.PRNGKey(0)


@pytest.fixture
def params(network: mz.nn.NeuralNetwork, random_key):
    return network.init(random_key)


class DummyVariableSource(VariableSource):
    def __init__(self, params) -> None:
        self._params = params

    def get_variables(self, names: Sequence[str]) -> List[types.NestedArray]:
        return [self._params]


@pytest.fixture
def variable_client(params):
    return VariableClient(DummyVariableSource(params=params), None)
