# %%
import contextlib
import collections
import copy
import dataclasses
import enum
import functools
import inspect
import types
import time
from typing import Any, Callable, Coroutine, Dict, List, NamedTuple, Optional, Union

import attr
import trio
import acme
import dm_env
import jax
import moozi as mz
import numpy as np
import open_spiel

import ray
import tree
from acme.wrappers import SinglePrecisionWrapper
from acme.wrappers.open_spiel_wrapper import OpenSpielWrapper
from moozi.utils import SimpleQueue

from link import link

ray.init(ignore_reinit_error=True)

# %%
def make_catch():
    env_columns, env_rows = 5, 5
    raw_env = open_spiel.python.rl_environment.Environment(
        f"catch(columns={env_columns},rows={env_rows})"
    )
    env = OpenSpielWrapper(raw_env)
    env = SinglePrecisionWrapper(env)
    env_spec = acme.specs.make_environment_spec(env)
    return env, env_spec


def make_tic_tac_toe():
    raw_env = open_spiel.python.rl_environment.Environment(f"tic_tac_toe")
    env = OpenSpielWrapper(raw_env)
    env = SinglePrecisionWrapper(env)
    env_spec = acme.specs.make_environment_spec(env)
    return env, env_spec


make_env = make_catch

# %%
@attr.s(auto_attribs=True)
class Artifact:
    # meta
    num_ticks: int = 0

    # environment
    env_state: dm_env.Environment = None
    timestep: dm_env.TimeStep = None
    to_play: int = -1
    action: int = -1

    # player
    last_frames: Optional[SimpleQueue] = None


# %%
@link
def print_timestep(timestep):
    print(timestep)


@link
def random_action_law(timestep: dm_env.TimeStep):
    if not timestep.last():
        legal_actions = timestep.observation[0].legal_actions
        random_action = np.random.choice(np.flatnonzero(legal_actions == 1))
        return {"action": random_action}
    else:
        return {"action": -1}


@link
@attr.s
class EnvironmentLaw:
    env_state = attr.ib()

    def __call__(self, timestep: dm_env.TimeStep, action: int):
        if timestep is None or timestep.last():
            timestep = self.env_state.reset()
        else:
            timestep = self.env_state.step([action])
        return {"timestep": timestep}


@link
def increment_tick(num_ticks):
    return {"num_ticks": num_ticks + 1}


# %%
# num_simulators = 10
# batch_size = 9


# async def process_batch(send_responses_channels, batch):
#     results = [(idx, c + 100) for idx, c in batch]
#     print(f"processed {len(batch)} tasks")

#     for idx, data in results:
#         await send_responses_channels[idx].send((idx, data))


# async def main():
#     async with trio.open_nursery() as nursery:
#         send_request_channel, receive_request_channel = trio.open_memory_channel(0)

#         send_response_channels, receive_response_channels = [], []
#         for i in range(num_simulators):
#             sender, receiver = trio.open_memory_channel(0)
#             send_response_channels.append(sender)
#             receive_response_channels.append(receiver)

#         nursery.start_soon(
#             batching_layer, receive_request_channel, send_response_channels
#         )

#         async with send_request_channel:
#             for i in range(num_simulators):
#                 nursery.start_soon(
#                     simulation_runner,
#                     i,
#                     send_request_channel.clone(),
#                     receive_response_channels[i],
#                 )

#     print("all done")


# trio.run(main)


trio.run(test_batching_layer)

# %%
# async def run(self):
#     async with trio.open_nursery() as nursery:

#         send_response_channels, receive_response_channels = [], []
#         for i in range(num_simulators):
#             sender, receiver = trio.open_memory_channel(0)
#             send_response_channels.append(sender)
#             receive_response_channels.append(receiver)

#         nursery.start_soon(
#             batching_layer, receive_request_channel, send_response_channels
#         )

#         async with send_request_channel:
#             for i in range(num_simulators):
#                 nursery.start_soon(
#                     simulation_runner,
#                     i,
#                     send_request_channel.clone(),
#                     receive_response_channels[i],
#                 )

#     print("all done")

# async def batching_layer(self):
#     print("batch processor starting")
#     async with contextlib.AsyncExitStack() as stack:
#         for context in [receive_request_channel, *send_responses_channels]:
#             await stack.enter_async_context(context)

#         batch = []
#         tick_timeframe = 1
#         while True:
#             try:
#                 mode = "move on"
#                 with trio.move_on_after(tick_timeframe):
#                     while True:
#                         data = await receive_request_channel.receive()
#                         batch.append(data)
#                         if len(batch) >= batch_size:
#                             mode = "skipped"
#                             break

#                 print(mode)
#                 await process_batch(send_responses_channels, batch)
#                 batch.clear()

#             except trio.EndOfChannel:
#                 assert len(batch) == 0
#                 break

#     print("batch processor closed")


# async def simulation_runner(idx, send_request_channel, receive_response_channel):
#     print("policy runner", idx, "starting")

#     async with send_request_channel, receive_response_channel:
#         counter = 0
#         while counter < 5:
#             await trio.sleep(1)
#             task = (idx, counter)
#             print(task, "requesting")
#             await send_request_channel.send(task)
#             result = await receive_response_channel.receive()
#             print(result, "done")
#             counter += 1
#     print("policy runner", idx, "closed")


# %%
num_universes = 10
num_ticks = 100


@link
@attr.s
class Planner:
    send_request = attr.ib()
    receive_response = attr.ib()
    close = attr.ib()

    async def __call__(self, timestep):
        await self.send_request(timestep)
        response = await self.receive_response()
        return response


@ray.remote
class InteractionManager:
    async def run(self):
        batching_layer = BatchingLayer(batch_size=10, tick_timeframe=1)

        def make_artifact():
            return Artifact()

        def make_planner():
            batching_layer
            return Planner()

        def make_laws():
            return [
                EnvironmentLaw(make_env()[0]),
                make_planner(),
                increment_tick,
            ]

        def make_universe():
            artifact = make_artifact()
            laws = make_laws()
            return Universe(artifact, laws)

        universes = [make_universe() for i in range(num_universes)]

        async with trio.open_nursery() as nursery:
            nursery.start_soon(batching_layer.run)

            async with batching_layer:
                for _ in range(num_ticks):
                    await self._tick_once(universes)

    # async def _tick_once(self, universes):
    #     async with trio.open_nursery() as nursery:
    #         for u in universes:
    #             nursery.start_soon(u.tick)


@attr.s
class Driver:
    def run(self):
        mgr = InteractionManager.remote()
        print(mgr)
        print(mgr.run.remote())


# %%
driver = Driver()
driver.run()

# %%
