import acme.jax.variable_utils
import acme.wrappers.open_spiel_wrapper
import jax
import moozi as mz
import open_spiel.python.rl_environment
import pytest
from acme.agents import agent as acme_agent
from acme.agents import replay as acme_replay
from acme.environment_loops.open_spiel_environment_loop import OpenSpielEnvironmentLoop
from jax.config import config

config.update("jax_disable_jit", True)


@pytest.fixture
def env():
    raw_env = open_spiel.python.rl_environment.Environment("catch")
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
    dim_image = env_spec.observations.observation.shape
    nn_spec = mz.nn.NeuralNetworkSpec(
        dim_image=dim_image, dim_repr=2, dim_action=dim_action
    )
    return mz.nn.get_network(nn_spec)


@pytest.fixture
def reverb_replay(env_spec):
    batch_size = 16
    n_steps = 5
    return acme_replay.make_reverb_prioritized_nstep_replay(
        env_spec, batch_size=batch_size, n_step=n_steps
    )


@pytest.fixture
def learner(network):
    params = network.init(jax.random.PRNGKey(0))
    return mz.learner.RandomNoiseLearner(params)


@pytest.fixture
def variable_client(learner):
    return acme.jax.variable_utils.VariableClient(learner, None)


def test_prior_actor(env, env_spec, network, reverb_replay, variable_client, learner):
    actor = mz.actor.PriorPolicyActor(
        environment_spec=env_spec,
        network=network,
        adder=reverb_replay.adder,
        variable_client=variable_client,
        random_key=jax.random.PRNGKey(0),
    )
    agent = acme_agent.Agent(
        actor=actor, learner=learner, min_observations=0, observations_per_step=1
    )
    loop = OpenSpielEnvironmentLoop(env, [agent])
    result = loop.run_episode()
    assert result
