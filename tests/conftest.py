from typing import List, Sequence
from acme.core import VariableSource
from acme.jax.variable_utils import VariableClient
import jax
import pytest
import open_spiel
import acme

from acme import types
import moozi as mz


def update_jax_config():
    jax.config.update("jax_disable_jit", True)
    jax.config.update("jax_platform_name", "cpu")


update_jax_config()


@pytest.fixture
def env():
    raw_env = open_spiel.python.rl_environment.Environment("catch(columns=5,rows=5)")
    env = acme.wrappers.open_spiel_wrapper.OpenSpielWrapper(raw_env)
    env = acme.wrappers.SinglePrecisionWrapper(env)
    return env


@pytest.fixture
def env_spec(env):
    env_spec = acme.specs.make_environment_spec(env)
    return env_spec


@pytest.fixture
def network(env_spec):
    dim_action = env_spec.actions.num_values
    dim_image = env_spec.observations.observation.shape[0]
    nn_spec = mz.nn.NeuralNetworkSpec(
        dim_image=dim_image,
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
