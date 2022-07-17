import chex
import jax.numpy as jnp
import numpy as np
import pytest
from acme.jax.utils import add_batch_dim
from acme.wrappers.open_spiel_wrapper import OLT
import dm_env

from moozi import TrainTarget
# from moozi.nn.loss import MuZeroLoss


# @pytest.fixture
# def train_target(env: dm_env.Environment, num_unroll_steps, num_stacked_frames):
#     timestep = env.reset()
#     olt: OLT = timestep.observation[0]
#     obs = olt.observation
#     chex.assert_shape(obs, env.observation_spec().observation.shape)

#     stacked_frames = np.repeat(obs, num_stacked_frames, axis=-1)
#     action = np.zeros((num_unroll_steps,))
#     action_probs = np.zeros((num_unroll_steps, env.action_spec().num_values))
#     n_step_return = np.zeros((num_unroll_steps,))
#     last_reward = np.zeros((num_unroll_steps,))
#     root_value = np.zeros((num_unroll_steps,))
#     weight = np.ones((1,))

#     target = TrainTarget(
#         frame=stacked_frames,
#         action=action,
#         n_step_return=n_step_return,
#         last_reward=last_reward,
#         action_probs=action_probs,
#         root_value=root_value,
#         importance_sampling_ratio=weight,
#     ).cast()

#     return add_batch_dim(target)


# def test_mcts_loss(model, params, state, train_target):
#     loss = MuZeroLoss(num_unroll_steps=2)
#     assert loss(model, params, state, train_target)
