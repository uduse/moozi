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
from acme.utils.tree_utils import unstack_sequence_fields
from acme.wrappers import SinglePrecisionWrapper, open_spiel_wrapper
from jax._src.numpy.lax_numpy import stack
from moozi import batching_layer
from moozi.batching_layer import BatchingClient, BatchingLayer
from moozi.link import UniverseAsync, link
from moozi.nn import NeuralNetwork
from moozi.policies.mcts_async import MCTSAsync
from moozi.policies.policy import PolicyFeed
from moozi.utils import SimpleBuffer, WallTimer, check_ray_gpu
from trio_asyncio import aio_as_trio

# from acme.wrappers.open_spiel_wrapper import OpenSpielWrapper
from acme_openspiel_wrapper import OpenSpielWrapper
from interactions import (
    Artifact,
    BatchInferenceServer,
    EnvironmentLaw,
    FrameStacker,
    InteractionManager,
    increment_tick,
    make_catch,
    make_env,
    set_legal_actions,
    wrap_up_episode,
)
from config import Config, get_config_proxy

logging.set_verbosity(logging.DEBUG)


ray.init(num_gpus=1, ignore_reinit_error=True)


# %%
def get_random_action_from_timestep(timestep: dm_env.TimeStep):
    if not timestep.last():
        legal_actions = timestep.observation[0].legal_actions
        random_action = np.random.choice(np.flatnonzero(legal_actions == 1))
        return {"action": random_action}
    else:
        return {"action": -1}


def setup(inf_server_handler) -> Tuple[List[BatchingLayer], List[UniverseAsync]]:
    num_universes = 50

    async def init_inf_remote(x):
        return await aio_as_trio(inf_server_handler.init_inf.remote(x))

    async def recurr_inf_remote(x):
        return await aio_as_trio(inf_server_handler.recurr_inf.remote(x))

    bl_init_inf = BatchingLayer(
        max_batch_size=25, process_fn=init_inf_remote, name="batching [init]"
    )
    bl_recurr_inf = BatchingLayer(
        max_batch_size=25, process_fn=recurr_inf_remote, name="batching [recurr]"
    )

    def make_artifact(index):
        return Artifact(universe_id=index)

    def make_planner():
        mcts = MCTSAsync(
            init_inf_fn=bl_init_inf.spawn_client().request,
            recurr_inf_fn=bl_recurr_inf.spawn_client().request,
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
                return dict(action=action)

        return planner

    def make_laws():
        return [
            EnvironmentLaw(make_env()[0]),
            FrameStacker(num_frames=1),
            set_legal_actions,
            make_planner(),
            wrap_up_episode,
            increment_tick,
        ]

    def make_universe(index):
        artifact = make_artifact(index)
        laws = make_laws()
        return UniverseAsync(artifact, laws)

    universes = [make_universe(i) for i in range(num_universes)]
    return [bl_init_inf, bl_recurr_inf], universes


def make_inference_server_handler():
    env_spec = make_env()[1]
    num_stacked_frames = 1
    dim_repr = 64
    dim_action = env_spec.actions.num_values
    frame_shape = env_spec.observations.observation.shape
    stacked_frame_shape = (num_stacked_frames,) + frame_shape
    nn_spec = mz.nn.NeuralNetworkSpec(
        stacked_frames_shape=stacked_frame_shape,
        dim_repr=dim_repr,
        dim_action=dim_action,
        repr_net_sizes=(128, 128),
        pred_net_sizes=(128, 128),
        dyna_net_sizes=(128, 128),
    )
    network = mz.nn.get_network(nn_spec)
    params = network.init(jax.random.PRNGKey(0))

    inf_server = ray.remote(num_gpus=0.5)(BatchInferenceServer).remote(network, params)
    return inf_server


# %%
inf_server = make_inference_server_handler()

# %%
mgr = InteractionManager()
mgr.setup(partial(setup, inf_server))
result = mgr.run(5)

# %%
result = mgr.run(5)

# %%
import inspect
from dataclasses import dataclass, field
from typing import (
    AsyncContextManager,
    Awaitable,
    Callable,
    ClassVar,
    ContextManager,
    Coroutine,
)
from absl import logging
import numpy as np
import contextlib
import attr
import trio

logging.set_verbosity(logging.INFO)


# @attr.s(auto_attribs=True, repr=False)
@dataclass
class BatchingClient:
    client_id: int
    # send_request: Callable[..., Awaitable]
    # receive_response: Callable[..., Awaitable]
    request: Callable[..., Awaitable]
    open_context: AsyncContextManager
