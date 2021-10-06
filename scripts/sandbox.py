# %%
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, Dict, List, NamedTuple, Optional, Union

import attr
import dm_env
import jax
import numpy as np
import ray
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
from moozi.utils import SimpleBuffer

from interactions import (
    InteractionManager,
    Artifact,
    FrameStacker,
    EnvironmentLaw,
    make_env,
    PlayerShell,
    increment_tick,
    wrap_up_episode,
)

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


def make_interaction_manager():
    init_ref_batching_layer = BatchingLayer(
        max_batch_size=3,
        process_fn=lambda batch: aio_as_trio(policy_server.process_batch.remote(batch)),
    )

    def make_artifact(index):
        return Artifact(universe_id=index)

    def make_player_shell():
        client = self.batching_layers.spawn_client()
        return PlayerShell(client)

    def make_laws():
        return [
            EnvironmentLaw(make_env()[0]),
            FrameStacker(num_frames=1),
            make_player_shell(),
            wrap_up_episode,
            increment_tick,
        ]

    def make_universe(index):
        artifact = make_artifact(index)
        laws = make_laws()
        return AsyncUniverse(artifact, laws)

    self.universes = [make_universe(i) for i in range(self.num_universes)]

    policy_server = ray.remote(InferenceServer).remote()
    return InteractionManager


@attr.s
class Driver:
    def run(self):
        policy_server = ray.remote(InferenceServer).remote()

        batching_layer = BatchingLayer(
            max_batch_size=3,
            process_fn=lambda batch: aio_as_trio(
                policy_server.process_batch.remote(batch)
            ),
        )

        mgr = ray.remote(InteractionManager).remote(
            batching_layers=[batching_layer], num_universes=5, num_ticks=100
        )

        ref = mgr.run.remote()
        return ref


# %%
@dataclass
class InferenceServer:
    network: NeuralNetwork
    params: Any = None
    random_key = jax.random.PRNGKey(0)

    def __post_init__(self):
        self.random_key, next_key = jax.random.split(self.random_key)
        self.params = self.network.init(next_key)

    def init_inf(self, frames):
        return self.network.initial_inference(frames)

    def recurr_inf(self, hidden_states, actions):
        return self.network.recurrent_inference(hidden_states, actions)


# %%
@dataclass
class StupidInferenceServer:
    def init_inf(self, frames):
        return np.ones_like(frames)

    def recurr_inf(self, hidden_states, actions):
        return np.ones_like(hidden_states)


# %%
logging.set_verbosity(logging.DEBUG)
inf_server = ray.remote(StupidInferenceServer).remote()

# %%
async def process_fn(batch):
    logging.debug("batch:" + str(batch))
    return await trio_asyncio.aio_as_trio(inf_server.init_inf.remote(batch))


batching_layer = BatchingLayer(max_batch_size=5, process_fn=process_fn)
clients = [batching_layer.spawn_client() for _ in range(10)]

# %%
async def run():
    async with trio.open_nursery() as n:
        n.start_soon(batching_layer.start_processing)

        async with trio.open_nursery() as nn:
            for c in clients:
                nn.start_soon(c.request, 0)

        await batching_layer.close()


# %%
trio_asyncio.run(run)
