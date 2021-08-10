import jax.numpy as jnp
import moozi as mz
import numpy as np
import pytest
from acme.jax.utils import add_batch_dim
from acme.wrappers.open_spiel_wrapper import OLT
from dm_env import Environment


@pytest.fixture
def train_target(env: Environment, num_unroll_steps, num_stacked_frames):
    timestep: OLT = env.reset()
    obs = timestep.observation[0].observation
    stacked_frames = np.repeat(obs[np.newaxis, :], num_stacked_frames, axis=0)
    action = np.full((num_unroll_steps,), 0)
    action_probs = np.zeros((num_unroll_steps, env.action_spec().num_values))
    value = np.zeros((num_unroll_steps,))
    last_reward = np.zeros((num_unroll_steps,))
    target = mz.replay.TrainTarget(
        stacked_frames=stacked_frames,
        action=action,
        value=value,
        last_reward=last_reward,
        action_probs=action_probs,
    ).cast()
    return add_batch_dim(target)


def test_mcts_loss(network, params, train_target):
    loss = mz.loss.MuZeroLoss(num_unroll_steps=2)
    assert loss(network, params, train_target)
