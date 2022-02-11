import time

import acme.jax.variable_utils
import acme.wrappers.open_spiel_wrapper
import dm_env
import jax
import moozi as mz
import pytest
from acme.agents import agent as acme_agent
from acme.agents import replay as acme_replay
from acme.environment_loops.open_spiel_environment_loop import OpenSpielEnvironmentLoop
from pytest_mock import MockerFixture


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
    signature = mz.replay.make_signature(env_spec, max_replay_size)
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
    adder = mz.replay.Adder(client)
