# %%
import asyncio
from collections import deque
import functools
import operator
import os
import pprint
import random
from dataclasses import InitVar, asdict, dataclass, field
from functools import _make_key, partial
from operator import itemgetter
from os import terminal_size
import time
from typing import (
    Any,
    Callable,
    Coroutine,
    Deque,
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

import attr
import chex
import dm_env

import jax
import jax.numpy as jnp
import moozi as mz
import numpy as np
import optax
import ray
import rlax
import tqdm
import tree
import trio
import trio_asyncio
from absl import logging
from acme import specs
from acme.utils.loggers import TerminalLogger
from acme.utils.tree_utils import stack_sequence_fields, unstack_sequence_fields
from acme.wrappers import SinglePrecisionWrapper, open_spiel_wrapper
from moozi import batching_layer
from moozi.batching_layer import BatchingClient, BatchingLayer
from moozi.link import UniverseAsync, link, Universe
from moozi.logging import JAXBoardLogger, JAXBoardStepData
from moozi.nn import NeuralNetwork
from moozi.policies.mcts_async import MCTSAsync
from moozi.policies.policy import PolicyFeed
from moozi.replay import StepSample, TrajectorySample, make_target_from_traj
from moozi.utils import SimpleBuffer, WallTimer, as_coroutine, check_ray_gpu
from trio_asyncio import aio_as_trio

# from acme.wrappers.open_spiel_wrapper import OpenSpielWrapper
from acme_openspiel_wrapper import OpenSpielWrapper
from config import Config
from sandbox_core import (
    Artifact,
    EnvironmentLaw,
    FrameStacker,
    InferenceServer,
    MetricsReporterActor,
    ParameterOptimizer,
    ReplayBuffer,
    RolloutWorker,
    TrajectoryOutputWriter,
    increment_tick,
    make_catch,
    make_env,
    output_reward,
    update_episode_stats,
    set_random_action_from_timestep,
)

logging.set_verbosity(logging.INFO)


ray.init(ignore_reinit_error=True)


# %%
def make_planner_law(init_inf_fn, recurr_inf_fn, dim_actions, num_simulations=10):
    mcts = MCTSAsync(
        init_inf_fn=init_inf_fn,
        recurr_inf_fn=recurr_inf_fn,
        num_simulations=num_simulations,
        dim_action=dim_actions,
    )

    @link
    async def planner(is_last, stacked_frames, legal_actions_mask):
        if not is_last:
            feed = PolicyFeed(
                stacked_frames=stacked_frames,
                legal_actions_mask=legal_actions_mask,
                random_key=None,
            )
            mcts_tree = await mcts(feed)
            action, _ = mcts_tree.select_child()

            action_probs = np.zeros((3,), dtype=np.float32)
            for a, visit_count in mcts_tree.get_children_visit_counts().items():
                action_probs[a] = visit_count
            action_probs /= np.sum(action_probs)
            return dict(action=action, action_probs=action_probs)

    return planner


def make_rollout_worker_batching_layers(worker_self: RolloutWorker):
    def batched_init_inf(list_of_stacked_frames):
        batch_size = len(list_of_stacked_frames)
        results = worker_self.init_inf_fn(
            worker_self.params, np.array(list_of_stacked_frames)
        )
        results = tree.map_structure(np.array, results)
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
        max_batch_size=config.num_rollout_universes_per_worker / 2,
        process_fn=batched_init_inf,
        name="batching [init]",
        batch_process_period=1e-2,
    )
    bl_recurr_inf = BatchingLayer(
        max_batch_size=config.num_rollout_universes_per_worker / 2,
        process_fn=batched_recurr_inf,
        name="batching [recurr]",
        batch_process_period=1e-2,
    )
    return bl_init_inf, bl_recurr_inf


def make_rollout_worker_universes(
    worker_self: RolloutWorker, config: Config
) -> List[UniverseAsync]:
    bl_init_inf, bl_recurr_inf = worker_self.batching_layers

    def make_rollout_universe(index):
        artifact = config.artifact_factory(index)
        planner_law = make_planner_law(
            bl_init_inf.spawn_client().request,
            bl_recurr_inf.spawn_client().request,
            dim_actions=3,
        )
        laws = [
            EnvironmentLaw(config.env_factory()),
            FrameStacker(num_frames=config.num_stacked_frames),
            planner_law,
            TrajectoryOutputWriter(),
            update_episode_stats,
            increment_tick,
        ]
        return UniverseAsync(artifact, laws)

    universes = [
        make_rollout_universe(i) for i in range(config.num_rollout_universes_per_worker)
    ]
    return universes


def make_evaluator_universes(evaluator_self, config: Config) -> List[UniverseAsync]:
    artifact = config.artifact_factory(-1)
    planner_law = make_planner_law(
        lambda x: evaluator_self.init_inf_fn_unbatched(evaluator_self.params, x),
        lambda x: evaluator_self.recurr_inf_fn_unbatched(
            evaluator_self.params, x[0], x[1]
        ),
    )
    laws = [
        EnvironmentLaw(config.env_factory()),
        FrameStacker(num_frames=config.num_stacked_frames),
        planner_law,
        output_reward,
        update_episode_stats,
        increment_tick,
    ]
    return [UniverseAsync(artifact, laws)]


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


# %%
def setup_config(config: Config):
    def make_artifact(index):
        return Artifact(universe_id=index)

    config.batch_size = 256
    config.discount = 0.99
    config.num_unroll_steps = 3
    config.num_td_steps = 100
    config.num_stacked_frames = 1

    config.lr = 2e-3

    config.artifact_factory = make_artifact
    config.env_factory = lambda: make_catch()[0]
    config.env_spec = make_catch()[1]

    config.replay_buffer_size = 10000

    config.dim_repr = 64

    config.num_epochs = 20
    config.num_ticks_per_epoch = 10
    config.num_updates_per_samples_added = 30
    config.num_rollout_workers = 8
    config.num_rollout_universes_per_worker = 100

    num_interactions = (
        config.num_epochs
        * config.num_ticks_per_epoch
        * config.num_rollout_workers
        * config.num_rollout_universes_per_worker
    )
    print(f"num_interactions: {num_interactions}")


def setup_config_debug(config: Config):
    def make_artifact(index):
        return Artifact(universe_id=index)

    config.batch_size = 8
    config.discount = 0.99
    config.num_unroll_steps = 3
    config.num_td_steps = 100
    config.num_stacked_frames = 1

    config.lr = 2e-3

    config.artifact_factory = make_artifact
    config.env_factory = lambda: make_catch()[0]
    config.env_spec = make_catch()[1]

    config.replay_buffer_size = 100

    config.dim_repr = 1

    config.num_epochs = 2
    config.num_ticks_per_epoch = 10
    config.num_updates_per_samples_added = 2
    config.num_rollout_workers = 1
    config.num_rollout_universes_per_worker = 2


# %%
config = Config()
if os.getenv("DEBUG_MOOZI") is not None:
    setup_config_debug(config)
else:
    setup_config(config)
pprint.pprint(asdict(config))

# %%
metrics_reporter = MetricsReporterActor.remote()

# %%
# inf_server = (
#     ray.remote(num_gpus=1)(InferenceServer)
#     .options(name="Inference Server", max_concurrency=1000)
#     .remote()
# )

# ray.get(
#     [
#         inf_server.set_network.remote(network),
#         inf_server.set_params.remote(params),
#         inf_server.set_loggers.remote(
#             lambda: [
#                 # TerminalLogger(label="Inference Server", print_fn=print),
#             ]
#         ),
#     ]
# )

# %%
param_opt = ray.remote(ParameterOptimizer).options(num_gpus=0.5).remote()

ray.get(
    [
        param_opt.setup.remote(partial(setup_param_opt, config=config)),
        param_opt.set_loggers.remote(
            lambda: [
                # TerminalLogger(label="Parameter Optimizer", print_fn=print),
                mz.logging.JAXBoardLogger(name="param_opt"),
            ]
        ),
        param_opt.log_stats.remote(),
    ]
)

# %%
replay_buffer = ray.remote(ReplayBuffer).remote(config)

# %%
rollout_workers = []
for _ in range(config.num_rollout_workers):
    worker = ray.remote(RolloutWorker).remote()
    worker.set_network.remote(param_opt.get_network.remote())
    worker.set_params.remote(param_opt.get_params.remote())
    worker.set_batching_layers.remote(make_rollout_worker_batching_layers)
    worker.set_universes.remote(partial(make_rollout_worker_universes, config=config))
    rollout_workers.append(worker)

# %%
# ray.get(worker.run.remote(10))


# %%
evaluator = ray.remote(RolloutWorker).options(name="Evaluator").remote()
evaluator.set_network.remote(param_opt.get_network.remote())
evaluator.set_params.remote(param_opt.get_params.remote())
evaluator.set_universes.remote(partial(make_evaluator_universes, config=config))


@ray.remote
def evaluation_post_process(output_buffer):
    return JAXBoardStepData(
        scalars=dict(last_run_avr_reward=np.mean(output_buffer)), histograms=dict()
    )


# %%
def evaluate(num_ticks):
    output_buffer = evaluator.run.remote(num_ticks)
    step_data = evaluation_post_process.remote(output_buffer)
    return metrics_reporter.report.remote(step_data)


# %%
@ray.remote
def print_result(result):
    print(result)


# %%
t = WallTimer()
t.start()

for epoch in range(config.num_epochs):
    print(f"Epochs: {epoch + 1} / {config.num_epochs}")
    evaluation_done = evaluate(num_ticks=50)
    samples = [w.run.remote(config.num_ticks_per_epoch) for w in rollout_workers]
    samples_added = [replay_buffer.add_samples.remote(s) for s in samples]
    while samples_added:
        _, samples_added = ray.wait(samples_added)
        for _ in range(config.num_updates_per_samples_added):
            batch = replay_buffer.get_batch.remote(config.batch_size)
            param_opt.update.remote(batch)

        for w in rollout_workers + [evaluator]:
            w.set_params.remote(param_opt.get_params.remote())

        # inf_server.set_params.remote(new_params)
        # inf_server.log_stats.remote()
        param_opt.log_stats.remote()

    # print_result.remote(evaluation_result)
    # print_result.remote(inf_server.get_stats.remote())
    # print_result.remote(param_opt.get_stats.remote())
    # ray.get(evaluation_done)

ray.get(evaluation_done)

t.end()
t.print()

# %%
