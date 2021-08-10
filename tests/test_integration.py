from acme.environment_loops.open_spiel_environment_loop import OpenSpielEnvironmentLoop
from acme.jax.variable_utils import VariableClient
from acme.utils.loggers.base import NoOpLogger
import jax
import optax
import moozi as mz

import acme
from acme.agents.agent import Agent


def test_integration(
    env,
    env_spec,
    network,
    num_unroll_steps,
    num_stacked_frames,
):
    seed = 0
    key = jax.random.PRNGKey(seed)
    max_episode_length = env.environment.environment.game.max_game_length()
    num_td_steps = 2
    batch_size = 2
    discount = 0.99

    optimizer = optax.adam(1e-3)

    reverb_replay = mz.replay.make_replay(
        env_spec,
        max_episode_length=max_episode_length,
        batch_size=batch_size,
        min_replay_size=1,
        max_replay_size=100,
        prefetch_size=2,
    )

    data_iterator = mz.replay.post_process_data_iterator(
        reverb_replay.data_iterator,
        batch_size,
        discount,
        num_unroll_steps,
        num_td_steps,
        num_stacked_frames,
    )

    weight_decay = 1e-4
    loss_fn = mz.loss.MuZeroLoss(
        num_unroll_steps=num_unroll_steps, weight_decay=weight_decay
    )
    learner = mz.learner.SGDLearner(
        network,
        loss_fn=loss_fn,
        optimizer=optimizer,
        data_iterator=data_iterator,
        random_key=key,
        loggers=[],
    )

    variable_client = VariableClient(learner, None)
    policy = mz.policies.MonteCarlo(
        network, variable_client, num_unroll_steps=num_unroll_steps
    )
    actor = mz.MuZeroActor(
        env_spec,
        policy,
        reverb_replay.adder,
        key,
        num_stacked_frames=num_stacked_frames,
        loggers=[],
    )
    agent = Agent(
        actor=actor, learner=learner, min_observations=10, observations_per_step=1
    )
    loop = OpenSpielEnvironmentLoop(
        environment=env, actors=[agent], logger=NoOpLogger()
    )

    for _ in range(10):
        assert loop.run_episode()
