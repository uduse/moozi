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
            return dict(action=action, root=mcts_tree)

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
@dataclass(repr=False)
class ReplayBuffer:
    store: List[Trajectory] = field(default_factory=list)

    def append(self, traj: Trajectory):
        self.store.append(traj)

    def get(self, num_samples=1):
        sample = random.sample(self.store, num_samples)
        if num_samples == 1:
            return sample
        else:
            return stack_sequence_fields(sample)


# %%
def setup(config: Config, rb) -> Tuple[List[BatchingLayer], List[UniverseAsync]]:
    def make_rollout_laws():
        return [
            EnvironmentLaw(config.env_factory()),
            set_legal_actions,
            FrameStacker(num_frames=config.num_stacked_frames),
            set_random_action_from_timestep,
            link(lambda: dict(action_probs=np.random.randn(3))),
            TrajectorySaver(
                lambda x: rb.append.remote(x),
                # lambda x: print(f"saving {tree.map_structure(np.shape, x)}")
            ),
            wrap_up_episode,
            increment_tick,
        ]

    def make_universe():
        artifact = config.artifact_factory(0)
        laws = make_rollout_laws()
        return UniverseAsync(artifact, laws)

    return [], [make_universe()]


# %%
rb = ray.remote(ReplayBuffer).remote()
mgr = InteractionManager()
mgr.setup(partial(setup, config=config, rb=rb))

# %%
mgr.run(1000)

# %%
x = ray.get(rb.get.remote(10))
xx = tree.map_structure(itemgetter(0), x)
print(tree.map_structure(np.shape, xx))

# %%
xt = make_target(
    sample=xx,
    start_idx=0,
    discount=1,
    num_unroll_steps=config.num_unroll_steps,
    num_td_steps=config.num_td_steps,
    num_stacked_frames=config.num_stacked_frames,
)
print(tree.map_structure(np.shape, xt))
# random_start = random.randrange( + 1)

# %%
class TrainingState(NamedTuple):
    params: Any
    opt_state: optax.OptState
    steps: int
    rng_key: jax.random.KeyArray


@dataclass(repr=False)
class Learner:
    network: NeuralNetwork
    params: Any

    loss_fn: mz.loss.LossFn
    optimizer: optax.GradientTransformation

    sgd_step_fn: Callable = field(init=False)
    state: TrainingState = field(init=False)

    def __post_init__(self):
        self.state = TrainingState(
            params=self.params,
            opt_state=self.optimizer.init(self.params),
            steps=0,
            rng_key=jax.random.PRNGKey(0),
        )
        self.sgd_step_fn = _make_sgd_step_fn(self.network, self.loss_fn, self.optimizer)

    def __call__(self, batch):
        self.state, step_data = self.sgd_step_fn(self.state, batch)
        print(step_data.scalars)


# %%
network, params = make_network_and_params(config)
loss_fn = mz.loss.MuZeroLoss(
    num_unroll_steps=config.num_unroll_steps, weight_decay=config.weight_decay
)
step_fn = _make_sgd_step_fn(network, loss_fn, config.optimizer_factory())
learner = Learner(
    network=network,
    params=params,
    loss_fn=loss_fn,
    optimizer=config.optimizer_factory(),
)

# %%
