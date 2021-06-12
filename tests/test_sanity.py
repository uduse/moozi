import os
import sys
import time

import acme.jax.variable_utils
import acme.wrappers.open_spiel_wrapper
import dm_env
import jax
import moozi as mz
import open_spiel.python.rl_environment
import pytest
import reverb
from acme.agents import agent as acme_agent
from acme.agents import replay as acme_replay
from acme.environment_loops.open_spiel_environment_loop import OpenSpielEnvironmentLoop
from pytest_mock import MockerFixture


# TODO: add as global test variants
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
def n_step_reverb_replay(env_spec):
    # NOTE: fixtures are copied for each test, maybe problematic for a shared replay buffer
    batch_size = 3
    n_steps = 5
    return acme_replay.make_reverb_prioritized_nstep_replay(
        env_spec, batch_size=batch_size, n_step=n_steps
    )


@pytest.fixture
def random_noise_learner(network: mz.nn.NeuralNetwork):
    params = network.init(jax.random.PRNGKey(0))
    return mz.learner.RandomNoiseLearner(params)


@pytest.fixture
def random_actor(n_step_reverb_replay):
    return mz.actor.RandomActor(n_step_reverb_replay.adder)


# @pytest.fixture
# def variable_client(learner):
#     return acme.jax.variable_utils.VariableClient(learner, None)


def test_prior_actor(
    env, env_spec, network, n_step_reverb_replay, random_noise_learner
):
    variable_client = acme.jax.variable_utils.VariableClient(random_noise_learner, None)
    actor = mz.actor.PriorPolicyActor(
        environment_spec=env_spec,
        network=network,
        adder=n_step_reverb_replay.adder,
        variable_client=variable_client,
        random_key=jax.random.PRNGKey(0),
    )
    agent = acme_agent.Agent(
        actor=actor,
        learner=random_noise_learner,
        min_observations=0,
        observations_per_step=1,
    )
    loop = OpenSpielEnvironmentLoop(env, [agent])
    result = loop.run_episode()
    assert result


def test_n_step_prior_policy_gradient_loss(
    env, network, n_step_reverb_replay, random_actor, random_noise_learner
):
    loop = OpenSpielEnvironmentLoop(env, [random_actor])
    loop.run(num_episodes=2)
    batch = next(n_step_reverb_replay.data_iterator)
    loss = mz.loss.NStepPriorVanillaPolicyGradientLoss()
    result = loss(network, random_noise_learner.params, batch)
    assert result


def test_logger(mocker: MockerFixture, tmp_path):
    spy = mocker.spy(mz.logging.JAXBoardLogger, "_write_now")
    logger = mz.logging.JAXBoardLogger("test", log_dir=tmp_path, time_delta=0.01)
    time.sleep(0.1)
    logger.write(mz.logging.JAXBoardStepData({}, {}))
    logger.write(mz.logging.JAXBoardStepData({}, {}))
    logger.close()
    spy.assert_called_once()


def test_replay(env, env_spec):
    max_replay_size = 1000
    signature = mz.replay.make_signature(env_spec)
    replay_table = reverb.Table(
        name="test",
        sampler=reverb.selectors.Fifo(),
        remover=reverb.selectors.Fifo(),
        max_size=max_replay_size,
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature=signature,
    )
    server = reverb.Server([replay_table], port=None)
    address = f"localhost:{server.port}"
    client = reverb.Client(address)
    adder = mz.replay.MooZiAdder(client)



    
    


# def test_add_search_stats():
#     start = mz.utils.make_moozi_observation(
#         env_timestep=dm_env.restart(0), root_value=-10, child_visits=[1, 2]
#     )
#     mid = mz.utils.make_moozi_observation(
#         env_timestep=dm_env.transition(
#             reward=1, observation={"action": 0, "image": [0, 1]}, discount=0.5
#         ),
#         root_value=30,
#         child_visits=[10, 20],
#     )
#     end = mz.utils.make_moozi_observation(
#         env_timestep=dm_env.termination(
#             reward=3, observation={"action": 0, "image": [3, 4]}
#         ),
#         root_value=40,
#         child_visits=[100, 200],
#     )
#     print(start)
#     # mz.utils.make_moozi_observation(end, end_search)
