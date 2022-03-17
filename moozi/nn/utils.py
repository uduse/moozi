from typing import List

import nptyping
import optax
import tree
from acme.utils.tree_utils import unstack_sequence_fields

from moozi.batching_layer import BatchingLayer
from moozi import Config, make_env, make_env_spec
from moozi.core import UniverseAsync, Tape
from moozi.laws import (
    EnvironmentLaw,
    FrameStacker,
    TrajectoryOutputWriter,
    increment_tick,
    make_policy_feed,
    update_episode_stats,
    output_last_step_reward,
)
from moozi.policy.mcts_async import make_async_planner_law
from moozi.rollout_worker import RolloutWorkerWithWeights
import numpy as np
import moozi as mz
import jax


def make_rollout_worker_batching_layers(self: RolloutWorkerWithWeights, config: Config):
    def batched_init_inf(list_of_stacked_frames):
        batch_size = len(list_of_stacked_frames)
        results = self.root_inf(self.params, np.array(list_of_stacked_frames))
        results = tree.map_structure(np.array, results)
        return unstack_sequence_fields(results, batch_size)

    def batched_recurr_inf(inputs):
        batch_size = len(inputs)
        hidden_states, actions = zip(*inputs)
        hidden_states = np.array(hidden_states)
        actions = np.array(actions)
        results = self.trans_inf(self.params, hidden_states, actions)
        results = tree.map_structure(np.array, results)
        return unstack_sequence_fields(results, batch_size)

    bl_init_inf = BatchingLayer(
        max_batch_size=config.num_rollout_universes_per_worker,
        process_fn=batched_init_inf,
        name="[batched_init_inf]",
        batch_process_period=1e-1,
    )
    bl_recurr_inf = BatchingLayer(
        max_batch_size=config.num_rollout_universes_per_worker,
        process_fn=batched_recurr_inf,
        name="[batched_recurr_inf]",
        batch_process_period=1e-1,
    )
    return bl_init_inf, bl_recurr_inf


def make_param_opt_properties(config):
    env_spec = make_env_spec(config.env)
    dim_action = env_spec.actions.num_values
    frame_shape = env_spec.observations.observation.shape
    stacked_frame_shape = (config.num_stacked_frames,) + frame_shape
    nn_spec = mz.nn.NNSpec(
        obs_rows=stacked_frame_shape,
        dim_repr=config.dim_repr,
        dim_action=dim_action,
        repr_net_sizes=(128, 128),
        pred_net_sizes=(128, 128),
        dyna_net_sizes=(128, 128),
    )
    network = mz.nn.make_model(nn_spec)
    params, state = network.init_network(jax.random.PRNGKey(0))
    loss_fn = mz.loss.MuZeroLoss(
        num_unroll_steps=config.num_unroll_steps, weight_decay=config.weight_decay
    )
    optimizer = optax.adam(config.lr)
    return network, params, state, loss_fn, optimizer


def obs_to_ascii(obs):
    tokens = []
    for row in obs.reshape(3, 9).T:
        tokens.append(int(np.argwhere(row == 1)))
    tokens = np.array(tokens).reshape(3, 3)
    tokens = [row.tolist() for row in tokens]

    s = ""
    for row in tokens:
        for ele in row:
            if ele == 0:
                s += "."
            elif ele == 1:
                s += "O"
            else:
                s += "X"
        s += "\n"
    return s


def action_probs_to_ascii(action_probs):
    s = ""
    for row in action_probs.reshape(3, 3):
        for ele in row:
            s += f"{ele:.2f} "
        s += "\n"
    return s
