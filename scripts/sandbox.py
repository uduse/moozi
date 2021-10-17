# %%
import functools
import operator
from dataclasses import InitVar, dataclass, field
from functools import _make_key, partial
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
from acme import specs
from acme.agents.replay import ReverbReplay

import attr
import dm_env
import jax
import moozi as mz
import numpy as np
import ray
import tree
import trio
import trio_asyncio
from absl import logging
from acme.utils.tree_utils import stack_sequence_fields, unstack_sequence_fields
from acme.wrappers import SinglePrecisionWrapper, open_spiel_wrapper
from jax._src.numpy.lax_numpy import stack
from moozi import batching_layer
from moozi.batching_layer import BatchingClient, BatchingLayer
from moozi.link import UniverseAsync, link
from moozi.nn import NeuralNetwork
from moozi.policies.mcts_async import MCTSAsync
from moozi.policies.policy import PolicyFeed
from moozi.replay import Trajectory
from moozi.utils import SimpleBuffer, WallTimer, check_ray_gpu, as_coroutine
from trio_asyncio import aio_as_trio

# from acme.wrappers.open_spiel_wrapper import OpenSpielWrapper
from acme_openspiel_wrapper import OpenSpielWrapper
from sandbox_core import (
    Artifact,
    InferenceServer,
    EnvironmentLaw,
    FrameStacker,
    InteractionManager,
    increment_tick,
    make_catch,
    make_env,
    set_legal_actions,
    set_random_action_from_timestep,
    wrap_up_episode,
)
from config import Config

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
            return dict(action=action, root=mcts_tree)

    return planner


@link
@dataclass
class TrajectorySaver:
    traj_save_fn: Callable
    buffer: list = field(default_factory=list)

    def __post_init__(self):
        self.traj_save_fn = as_coroutine(self.traj_save_fn)

    async def __call__(self, timestep, stacked_frames, action, value, action_probs):

        step = Trajectory(
            frame=stacked_frames,
            reward=timestep.reward,
            is_first=timestep.first(),
            is_last=timestep.last(),
            action=action,
            root_value=value,
            action_probs=action_probs,
        ).cast()

        self.buffer.append(step)

        if timestep.last():
            final_traj = stack_sequence_fields(self.buffer)
            self.buffer.clear()
            await self.traj_save_fn(final_traj)


def setup(
    config: Config, inf_server_handler
) -> Tuple[List[BatchingLayer], List[UniverseAsync]]:

    init_inf_remote = InferenceServer.make_init_inf_remote_fn(inf_server_handler)
    recurr_inf_remote = InferenceServer.make_recurr_inf_remote_fn(inf_server_handler)

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


def make_inference_server_handler(config: Config):
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

    inf_server = ray.remote(num_gpus=0.5)(InferenceServer).remote(network, params)
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
config = Config()
setup_config(config)

# %%
inf_server = make_inference_server_handler(config)

# %%
mgr = InteractionManager()
mgr.setup(lambda: setup(config, inf_server))
result = mgr.run(5)

# %%
print(result[0].num_ticks)

# %%
result = mgr.run(5)

# %%
def convert(timestep):
    timestep = timestep._replace(observation=timestep.observation[0])
    frame = timestep.observation.observation.astype(np.float32)
    if timestep.reward is None:
        reward = np.float32(0)
    else:
        reward = np.float32(timestep.reward).squeeze()
    # legal_actions_mask = timestep.observation[0].legal_actions.astype(np.bool8)
    is_first = np.bool8(timestep.first())
    is_last = np.bool8(timestep.last())
    # return Observation(frame, reward, legal_actions_mask, is_first, is_last)
    return replay.Observation(frame, reward, is_first, is_last)


# %%
sampler = Sampler.remote(replay_factory)


# %%
r = replay_factory()

# %%
env = config.env_factory()

# %%
timestep = env.reset()
r.adder.add_first(convert(timestep))
while not timestep.last():
    timestep = env.step([0])
    r.adder.add(
        replay.Reflection(
            np.array(0, dtype=np.int32),
            np.array(0.0, dtype=np.float32),
            np.array([1, 2, 3], dtype=np.float32),
        ),
        convert(timestep),
    )

# %%
ref = sampler.sample.remote()

# %%
ray.get(ref, timeout=3)


# %%
def setup(config: Config) -> Tuple[List[BatchingLayer], List[UniverseAsync]]:
    def make_rollout_laws():
        return [
            EnvironmentLaw(config.env_factory()),
            set_legal_actions,
            FrameStacker(num_frames=config.num_stacked_frames),
            set_random_action_from_timestep,
            TrajectorySaver(lambda x: print(f"saving {x}")),
            wrap_up_episode,
            increment_tick,
        ]

    def make_universe():
        artifact = config.artifact_factory(0)
        laws = make_rollout_laws()
        return UniverseAsync(artifact, laws)

    return [], [make_universe()]


# %%
mgr = InteractionManager()
mgr.setup(partial(setup, config))

# %%
mgr.run(5)
# %%
