from typing import List

import nptyping
import optax
import tree
from acme.utils.tree_utils import unstack_sequence_fields

from moozi.batching_layer import BatchingLayer
from moozi.config import Config
from moozi.env import make_env
from moozi.laws import (
    EnvironmentLaw,
    FrameStacker,
    TrajectoryOutputWriter,
    increment_tick,
    set_policy_feed,
    update_episode_stats,
    output_reward,
)
from moozi.link import UniverseAsync
from moozi.materia import Materia
from moozi.policy.mcts_async import make_async_planner_law
from moozi.raw_env_factory import make_env_spec
from moozi.rollout_worker import RolloutWorkerWithWeights
import numpy as np
import moozi as mz
import jax


def make_rollout_worker_batching_layers(
    config: Config, worker_self: RolloutWorkerWithWeights
):
    def batched_init_inf(list_of_stacked_frames):
        batch_size = len(list_of_stacked_frames)
        results = worker_self.init_inf_fn(
            worker_self.params, np.array(list_of_stacked_frames)
        )
        results = tree.map_structure(nptyping.array, results)
        return unstack_sequence_fields(results, batch_size)

    def batched_recurr_inf(inputs):
        batch_size = len(inputs)
        hidden_states, actions = zip(*inputs)
        hidden_states = np.array(hidden_states)
        actions = np.array(actions)
        results = worker_self.recurr_inf_fn(worker_self.params, hidden_states, actions)
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


def make_rollout_worker_universes(
    worker_self: RolloutWorkerWithWeights, config: Config
) -> List[UniverseAsync]:
    assert len(worker_self.batching_layers) == 2
    bl_init_inf, bl_recurr_inf = worker_self.batching_layers

    def make_rollout_universe(index):
        materia = Materia(-1)
        planner_law = make_async_planner_law(
            bl_init_inf.spawn_client().request,
            bl_recurr_inf.spawn_client().request,
            dim_actions=3,
        )
        laws = [
            EnvironmentLaw(make_env(config.env)),
            FrameStacker(num_frames=config.num_stacked_frames),
            planner_law,
            TrajectoryOutputWriter(),
            update_episode_stats,
            increment_tick,
        ]
        return UniverseAsync(materia, laws)

    universes = [
        make_rollout_universe(i) for i in range(config.num_rollout_universes_per_worker)
    ]
    return universes


def make_evaluator_universes(evaluator_self, config: Config) -> List[UniverseAsync]:
    materia = Materia(-1)
    planner_law = make_async_planner_law(
        lambda x: evaluator_self.init_inf_fn_unbatched(evaluator_self.params, x),
        lambda x: evaluator_self.recurr_inf_fn_unbatched(
            evaluator_self.params, x[0], x[1]
        ),
        dim_actions=3,
    )
    laws = [
        EnvironmentLaw(make_env(config.env)),
        FrameStacker(num_frames=config.num_stacked_frames),
        set_policy_feed,
        planner_law,
        output_reward,
        update_episode_stats,
        increment_tick,
    ]
    return [UniverseAsync(materia, laws)]


def setup_param_opt(config):
    dim_action = config.env_spec.actions.num_values
    frame_shape = config.env_spec.observations.observation.shape
    stacked_frame_shape = (config.num_stacked_frames,) + frame_shape
    nn_spec = mz.nn.NeuralNetworkSpec(
        stacked_frames_shape=stacked_frame_shape,
        dim_repr=config.dim_repr,
        dim_action=dim_action,
        repr_net_sizes=(128, 128),
        pred_net_sizes=(128, 128),
        dyna_net_sizes=(128, 128),
    )
    network = mz.nn.get_network(nn_spec)
    params = network.init(jax.random.PRNGKey(0))
    loss_fn = mz.loss.MuZeroLoss(
        num_unroll_steps=config.num_unroll_steps, weight_decay=config.weight_decay
    )
    optimizer = optax.adam(config.lr)
    return network, params, loss_fn, optimizer
