# %%
import functools
import operator
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
from jax._src.numpy.lax_numpy import stack
from moozi import batching_layer
from moozi.batching_layer import BatchingClient, BatchingLayer
from moozi.link import UniverseAsync, link
from moozi.logging import JAXBoardLogger
from moozi.nn import NeuralNetwork
from moozi.policies.mcts_async import MCTSAsync
from moozi.policies.policy import PolicyFeed
from moozi.replay import Trajectory, make_target
from moozi.utils import SimpleBuffer, WallTimer, as_coroutine, check_ray_gpu
from trio_asyncio import aio_as_trio

# from acme.wrappers.open_spiel_wrapper import OpenSpielWrapper
from acme_openspiel_wrapper import OpenSpielWrapper
from config import Config
from sandbox_core import (
    Artifact,
    EnvironmentLaw,
    EpisodeStatsReporter,
    FrameStacker,
    InferenceServer,
    ReplayBuffer,
    RolloutWorker,
    increment_tick,
    make_catch,
    make_env,
    save_episode_stats,
    set_legal_actions,
    set_random_action_from_timestep,
)

logging.set_verbosity(logging.INFO)


ray.init(ignore_reinit_error=True)


# %%
def make_planner_law(init_inf_fn, recurr_inf_fn):
    mcts = MCTSAsync(
        init_inf_fn=init_inf_fn,
        recurr_inf_fn=recurr_inf_fn,
        num_simulations=10,
        dim_action=3,
    )

    @link
    async def planner(timestep, stacked_frames, legal_actions_mask):
        if not timestep.last():
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


@link
@dataclass
class TrajectorySaver:
    traj_save_fn: Callable
    buffer: list = field(default_factory=list)

    def __post_init__(self):
        self.traj_save_fn = as_coroutine(self.traj_save_fn)

    async def __call__(self, timestep, obs, action, root_value, action_probs):

        # hack for open_spiel reward structure
        if timestep.reward is None:
            reward = 0.0
        else:
            reward = timestep.reward[0]
        step = Trajectory(
            frame=obs,
            reward=reward,
            is_first=timestep.first(),
            is_last=timestep.last(),
            action=action,
            root_value=root_value,
            action_probs=action_probs,
        ).cast()

        self.buffer.append(step)

        if timestep.last():
            final_traj = stack_sequence_fields(self.buffer)
            self.buffer.clear()
            await self.traj_save_fn(final_traj)


def rollout_worker_setup(
    config: Config, inf_server_handle, replay_handle
) -> Tuple[List[BatchingLayer], List[UniverseAsync]]:
    init_inf_remote = InferenceServer.make_init_inf_remote_fn(inf_server_handle)
    recurr_inf_remote = InferenceServer.make_recurr_inf_remote_fn(inf_server_handle)

    bl_init_inf = BatchingLayer(
        max_batch_size=config.num_rollout_universes_per_worker / 2,
        process_fn=init_inf_remote,
        name="batching [init]",
        batch_process_period=1e-2,
    )
    bl_recurr_inf = BatchingLayer(
        max_batch_size=config.num_rollout_universes_per_worker / 2,
        process_fn=recurr_inf_remote,
        name="batching [recurr]",
        batch_process_period=1e-2,
    )

    def make_rollout_universe(index):
        artifact = config.artifact_factory(index)
        planner_law = make_planner_law(
            bl_init_inf.spawn_client().request, bl_recurr_inf.spawn_client().request
        )
        laws = [
            EnvironmentLaw(config.env_factory()),
            set_legal_actions,
            FrameStacker(num_frames=config.num_stacked_frames),
            planner_law,
            TrajectorySaver(lambda x: replay_handle.add_traj.remote(x)),
            increment_tick,
        ]
        return UniverseAsync(artifact, laws)

    universes = [
        make_rollout_universe(i) for i in range(config.num_rollout_universes_per_worker)
    ]
    return [bl_init_inf, bl_recurr_inf], universes


def evaluation_worker_setup(
    config: Config, inf_server_handle
) -> Tuple[List[BatchingLayer], List[UniverseAsync]]:
    def make_evaluator_universe():
        artifact = config.artifact_factory(-1)
        planner_law = make_planner_law(
            lambda x: ray.get(inf_server_handle.init_inf.remote([x]))[0],
            lambda x: ray.get(inf_server_handle.recurr_inf.remote([x]))[0],
        )
        laws = [
            EnvironmentLaw(config.env_factory()),
            set_legal_actions,
            FrameStacker(num_frames=config.num_stacked_frames),
            planner_law,
            save_episode_stats,
            EpisodeStatsReporter(mz.logging.JAXBoardLogger(name="evaluator")),
            increment_tick,
        ]
        return UniverseAsync(artifact, laws)

    return [], [make_evaluator_universe()]


def make_network_and_params(config):
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
    return network, params


def make_inference_server_handle(config: Config):
    network, params = make_network_and_params(config)
    loss_fn = mz.loss.MuZeroLoss(
        num_unroll_steps=config.num_unroll_steps, weight_decay=config.weight_decay
    )
    optimizer = optax.adam(config.lr)
    inf_server = (
        ray.remote(num_gpus=1)(InferenceServer)
        .options(name="Inference Server", max_concurrency=5)
        .remote(network, params, loss_fn, optimizer)
    )
    inf_server.set_loggers.remote(
        lambda: [
            # TerminalLogger(),
            mz.logging.JAXBoardLogger(name="inf_server", time_delta=10.0),
        ]
    )
    return inf_server


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

    config.replay_buffer_size = 100000

    config.dim_repr = 64

    config.num_epochs = 500
    config.num_ticks_per_epoch = 30
    config.num_updates_per_epoch = 30
    config.num_rollout_workers = 1
    config.num_rollout_universes_per_worker = 100

    # config.num_ticks = int(
    #     250_000 / (config.num_rollout_workers * config.num_rollout_universes_per_worker)
    # )


# %%
config = Config()
setup_config(config)
pprint.pprint(asdict(config))

# %%
inf_server_handle = make_inference_server_handle(config)
replay_handle = ray.remote(ReplayBuffer).remote(config)

mgrs = []
for _ in range(config.num_rollout_workers):
    mgr = ray.remote(RolloutWorker).remote()
    mgr.setup.remote(
        partial(
            rollout_worker_setup,
            config=config,
            inf_server_handle=inf_server_handle,
            replay_handle=replay_handle,
        )
    )
    mgrs.append(mgr)

mgr = ray.remote(RolloutWorker).options(name="Evaluator").remote()
mgr.setup.remote(
    partial(evaluation_worker_setup, config=config, inf_server_handle=inf_server_handle)
)
mgrs.append(mgr)

# %%
for i in tqdm.tqdm(range(config.num_epochs)):
    tasks = []
    for mgr in mgrs:
        rollout_task = mgr.run.remote(config.num_ticks_per_epoch)
        tasks.append(rollout_task)

    for _ in range(config.num_updates_per_epoch):
        batch = replay_handle.get_batch.remote(config.batch_size)
        update_task = inf_server_handle.update.remote(batch)
        tasks.append(update_task)

    ray.get(tasks)

# # %%
# for mgr in mgrs:
#     mgr.run.remote(config.num_ticks)

# time.sleep(10)
# for i in range(config.num_epochs):
#     time.sleep(0.1)
#     batch = replay_handle.get_batch.remote(config.batch_size)
#     updated = inf_server_handle.update.remote(batch)
#     ray.get(updated)
