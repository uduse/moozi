# %%
import functools
import operator
import random
from dataclasses import InitVar, dataclass, field
from functools import _make_key, partial
from operator import itemgetter
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
import tree
import trio
import trio_asyncio
from absl import logging
from acme import specs
from acme.utils.tree_utils import stack_sequence_fields, unstack_sequence_fields
from acme.wrappers import SinglePrecisionWrapper, open_spiel_wrapper
from jax._src.numpy.lax_numpy import stack
from moozi import batching_layer
from moozi.batching_layer import BatchingClient, BatchingLayer
from moozi.link import UniverseAsync, link
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
    FrameStacker,
    InferenceServer,
    InteractionManager,
    increment_tick,
    make_catch,
    make_env,
    set_legal_actions,
    set_random_action_from_timestep,
    wrap_up_episode,
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


def setup(
    config: Config, inf_server_handle, replay_handle
) -> Tuple[List[BatchingLayer], List[UniverseAsync]]:

    init_inf_remote = InferenceServer.make_init_inf_remote_fn(inf_server_handle)
    recurr_inf_remote = InferenceServer.make_recurr_inf_remote_fn(inf_server_handle)

    bl_init_inf = BatchingLayer(
        max_batch_size=25,
        process_fn=init_inf_remote,
        name="batching [init]",
        batch_process_period=0.001,
    )
    bl_recurr_inf = BatchingLayer(
        max_batch_size=25,
        process_fn=recurr_inf_remote,
        name="batching [recurr]",
        batch_process_period=0.001,
    )

    def make_rollout_laws():
        return [
            EnvironmentLaw(config.env_factory()),
            set_legal_actions,
            FrameStacker(num_frames=config.num_stacked_frames),
            make_planner_law(
                bl_init_inf.spawn_client().request, bl_recurr_inf.spawn_client().request
            ),
            TrajectorySaver(
                lambda x: replay_handle.append.remote(x),
                # lambda x: print(f"saving {tree.map_structure(np.shape, x)}")
            ),
            wrap_up_episode,
            increment_tick,
        ]

    def make_rollout_universe(index):
        artifact = config.artifact_factory(index)
        laws = make_rollout_laws()
        return UniverseAsync(artifact, laws)

    rollout_universes = [
        make_rollout_universe(i) for i in range(config.num_rollout_universes)
    ]

    return [bl_init_inf, bl_recurr_inf], rollout_universes


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


def make_inference_server_handler(config: Config):
    network, params = make_network_and_params(config)
    loss_fn = mz.loss.MuZeroLoss(
        num_unroll_steps=config.num_unroll_steps, weight_decay=config.weight_decay
    )
    optimizer = optax.adam(config.lr)
    inf_server = ray.remote(num_gpus=1)(InferenceServer).remote(
        network, params, loss_fn, optimizer
    )
    return inf_server


# %%
def setup_config(config: Config):
    def make_artifact(index):
        return Artifact(universe_id=index)

    config.artifact_factory = make_artifact
    config.env_factory = lambda: make_catch()[0]
    config.env_spec = make_catch()[1]

    config.num_rollout_universes = 5


# %%
@dataclass(repr=False)
class ReplayBuffer:
    config: Config

    store: List[Trajectory] = field(default_factory=list)

    def append(self, traj: Trajectory):
        self.store.append(traj)

    def get(self, num_samples=1):
        trajs = random.sample(self.store, num_samples)
        batch = []
        for traj in trajs:
            random_start_idx = random.randrange(len(traj.reward))
            target = make_target(
                traj,
                start_idx=random_start_idx,
                discount=1.0,
                num_unroll_steps=self.config.num_unroll_steps,
                num_td_steps=self.config.num_td_steps,
                num_stacked_frames=self.config.num_stacked_frames,
            )
            batch.append(target)
        return stack_sequence_fields(batch)


# %%
config = Config()
setup_config(config)

# %%
inf_server_handle = make_inference_server_handler(config)
replay_handle = ray.remote(ReplayBuffer).remote(config)
mgr = InteractionManager()
mgr.setup(
    partial(
        setup,
        config=config,
        inf_server_handle=inf_server_handle,
        replay_handle=replay_handle,
    )
)

# %%
results = mgr.run(500)

# %%
for _ in range(100):
    loss = ray.get(
        inf_server_handle.update.remote(replay_handle.get.remote(50))
    ).scalars["loss"]
    print(loss)
