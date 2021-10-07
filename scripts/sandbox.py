# %%
from dataclasses import InitVar, dataclass, field
import functools
from functools import partial
import operator
from typing import Any, Callable, Coroutine, Dict, List, NamedTuple, Optional, Union
from acme.utils.tree_utils import unstack_sequence_fields

import attr
import dm_env
import jax
import numpy as np
import ray
import tree
import trio
import trio_asyncio
from trio_asyncio import aio_as_trio
from absl import logging
from acme.wrappers import SinglePrecisionWrapper
from acme.wrappers.open_spiel_wrapper import OpenSpielWrapper
from jax._src.numpy.lax_numpy import stack
from moozi import batching_layer
from moozi.batching_layer import BatchingClient, BatchingLayer
from moozi.link import AsyncUniverse, link
from moozi.nn import NeuralNetwork
from moozi.policies.mcts_async import MCTSAsync
from moozi.utils import SimpleBuffer
import moozi as mz

from interactions import (
    InteractionManager,
    Artifact,
    FrameStacker,
    EnvironmentLaw,
    make_catch,
    make_env,
    PlayerShell,
    increment_tick,
    wrap_up_episode,
    BatchInferenceServer,
)
from scripts import interactions

logging.set_verbosity(logging.DEBUG)

ray.init(ignore_reinit_error=True)


# %%
def get_random_action_from_timestep(timestep: dm_env.TimeStep):
    if not timestep.last():
        legal_actions = timestep.observation[0].legal_actions
        random_action = np.random.choice(np.flatnonzero(legal_actions == 1))
        return {"action": random_action}
    else:
        return {"action": -1}


def setup():
    env, env_spec = make_catch()

    num_stacked_frames = 1
    num_universes = 10
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

    inf_server = ray.remote(BatchInferenceServer).remote(network, params)

    async def init_inf_remote(x):
        return await aio_as_trio(inf_server.init_inf.remote(x))

    async def recurr_inf_remote(x):
        return await aio_as_trio(inf_server.recurr_inf.remote(x))

    bl_init_inf = BatchingLayer(max_batch_size=3, process_fn=init_inf_remote)
    bl_recurr_inf = BatchingLayer(max_batch_size=3, process_fn=recurr_inf_remote)

    def make_artifact(index):
        return Artifact(universe_id=index)

    def make_planner():
        mcts = MCTSAsync(
            init_inf_fn=bl_init_inf.spawn_client().request,
            recurr_inf_fn=bl_recurr_inf.spawn_client().request,
            num_simulations=10,
            dim_action=dim_action,
        )

        @link
        async def planner(stacked_frames):
            mcts_tree = await mcts(stacked_frames)
            action, _ = mcts_tree.select_child()
            return dict(action=action)

        return planner

    def make_laws():
        return [
            EnvironmentLaw(make_env()[0]),
            FrameStacker(num_frames=1),
            make_planner(),
            wrap_up_episode,
            increment_tick,
        ]

    def make_universe(index):
        artifact = make_artifact(index)
        laws = make_laws()
        return AsyncUniverse(artifact, laws)

    universes = [make_universe(i) for i in range(num_universes)]

    mgr = ray.remote(InteractionManager).remote(
        batching_layers=[bl_init_inf, bl_recurr_inf], universes=universes
    )
    return mgr


class Driver:
    def run(self):
        mgr = setup()
        ref = mgr.run.remote()
        return ref


# %%
driver = Driver()
driver.run()