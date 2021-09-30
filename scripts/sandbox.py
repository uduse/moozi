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

# import ray
import tree
from acme.wrappers import SinglePrecisionWrapper
from acme.wrappers.open_spiel_wrapper import OpenSpielWrapper
from moozi.utils import SimpleQueue

# %%
# ray.init(ignore_reinit_error=True)

# %%
def env_factory():
    env_columns, env_rows = 5, 5
    raw_env = open_spiel.python.rl_environment.Environment(
        f"catch(columns={env_columns},rows={env_rows})"
    )
    env = OpenSpielWrapper(raw_env)
    env = SinglePrecisionWrapper(env)
    env_spec = acme.specs.make_environment_spec(env)
    return env, env_spec


# def make_env():
#     raw_env = open_spiel.python.rl_environment.Environment(f"tic_tac_toe")
#     env = OpenSpielWrapper(raw_env)
#     env = SinglePrecisionWrapper(env)
#     env_spec = acme.specs.make_environment_spec(env)
#     return env, env_spec


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
@attr.s(auto_attribs=True)
class Link:
    callable_obj: Any
    to_read: Union[List[str], str] = "auto"
    to_write: Union[List[str], str] = "auto"

    async def __call__(self, artifact: object):
        keys_to_read = self._get_keys_to_read(artifact)
        artifact_window = Link._read_artifact(artifact, keys_to_read)
        if inspect.iscoroutine(self.callable_obj):
            updates = await self.callable_obj(**artifact_window)
        else:
            updates = self.callable_obj(**artifact_window)
        if not updates:
            updates = {}
        self._validate_updates(artifact, updates)
        Link._update_artifact(artifact, updates)
        return updates

    @staticmethod
    def _read_artifact(artifact, keys_to_read: List[str]):
        return {key: getattr(artifact, key) for key in keys_to_read}

    @staticmethod
    def _update_artifact(artifact, updates: Dict[str, Any]):
        for key, val in updates.items():
            setattr(artifact, key, val)

    @staticmethod
    def _artifact_has_keys(artifact, keys: List[str]) -> bool:
        return set(keys) <= set(artifact.__dict__.keys())

    def _validate_updates(self, artifact, updates):
        if not Link._artifact_has_keys(artifact, list(updates.keys())):
            raise ValueError

        if self.to_write == "auto":
            pass
        elif isinstance(self.to_write, list):
            update_nothing = (not self.to_write) and (not updates)
            if update_nothing:
                pass
            elif self.to_write != list(updates.keys()):
                raise ValueError("write_view keys mismatch.")
        else:
            raise ValueError("`to_write` type not accepted.")

    def _wrapped_func_keys(self):
        return set(inspect.signature(self.callable_obj).parameters.keys())

    def _get_keys_to_read(self, artifact):
        if self.to_read == "auto":
            keys = self._wrapped_func_keys()
            keys = keys - {"self"}  ## TODO?
            if not Link._artifact_has_keys(artifact, keys):
                raise ValueError(f"{str(keys)} not in {str(artifact.__dict__.keys())})")
        elif isinstance(self.to_read, list):
            keys = self.to_read
        else:
            raise ValueError("`to_read` type not accepted.")
        return keys


# %%
@attr.s
class LinkClassWrapper:
    class_: type = attr.ib()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return link(self.class_(*args, **kwargs))


# %%
def link(*args, **kwargs):
    if len(args) == 1 and not kwargs and inspect.isclass(args[0]):
        return LinkClassWrapper(args[0])
    elif len(args) == 1 and not kwargs and callable(args[0]):
        func = args[0]
        return Link(func, to_read="auto", to_write="auto")
    else:
        func = functools.partial(Link, *args, **kwargs)
        return func


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


@attr.s
class Universe:
    artifact = attr.ib()
    laws = attr.ib()

    async def tick(self):
        for law in self.laws:
            await law(self.artifact)


# %%
def make_artifact():
    artifact = Artifact()
    artifact.env_state = make_env()[0]
    return artifact


def make_laws():
    return [
        EnvironmentLaw(make_env()[0]),
        random_action_law,
        increment_tick,
    ]


# %%
async def main():
    univeres = [Universe(make_artifact(), make_laws()) for _ in range(10)]
    for u in univeres:
        await u.tick()
    # n.start_soon(u.tick)


trio.run(main)

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

# %%
class BatchingLayer:
    def __init__(self, batch_size, tick_timeframe):
        self._batch_size = batch_size
        self._tick_timeframe = tick_timeframe

        (
            self._send_request_channel,
            self._receive_request_channel,
        ) = trio.open_memory_channel(0)

    async def run(self):
        async with trio.open_nursery() as nursery:

            send_response_channels, receive_response_channels = [], []
            for i in range(num_simulators):
                sender, receiver = trio.open_memory_channel(0)
                send_response_channels.append(sender)
                receive_response_channels.append(receiver)

            nursery.start_soon(
                batching_layer, receive_request_channel, send_response_channels
            )

            async with send_request_channel:
                for i in range(num_simulators):
                    nursery.start_soon(
                        simulation_runner,
                        i,
                        send_request_channel.clone(),
                        receive_response_channels[i],
                    )

        print("all done")

    async def batching_layer(self):
        print("batch processor starting")
        async with contextlib.AsyncExitStack() as stack:
            for context in [receive_request_channel, *send_responses_channels]:
                await stack.enter_async_context(context)

            batch = []
            tick_timeframe = 1
            while True:
                try:
                    mode = "move on"
                    with trio.move_on_after(tick_timeframe):
                        while True:
                            data = await receive_request_channel.receive()
                            batch.append(data)
                            if len(batch) >= batch_size:
                                mode = "skipped"
                                break

                    print(mode)
                    await process_batch(send_responses_channels, batch)
                    batch.clear()

                except trio.EndOfChannel:
                    assert len(batch) == 0
                    break

        print("batch processor closed")


# %%
@attr.s
class BatchingProxy:
    _send_request_channel = attr.ib()
    _receive_response_channel = attr.ib()

    async def send(self):
        pass


class Simulation:
    def __init__(self, sender, receiver):
        self._sender = sender
        self._receiver = receiver

    async def run(self):
        await trio.sleep(0.1)
        self._sender


async def simulation_runner(idx, send_request_channel, receive_response_channel):
    print("policy runner", idx, "starting")

    async with send_request_channel, receive_response_channel:
        counter = 0
        while counter < 5:
            await trio.sleep(1)
            task = (idx, counter)
            print(task, "requesting")
            await send_request_channel.send(task)
            result = await receive_response_channel.receive()
            print(result, "done")
            counter += 1
    print("policy runner", idx, "closed")


async def main():
    num_universes = 10
    num_ticks = 100
    batching_layer = BatchingLayer(batch_size=10, tick_timeframe=1)

    def make_artifact():
        return Artifact()

    def make_laws():
        return [
            EnvironmentLaw(make_env()[0]),
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
                for u in universes:
                    nursery.start_soon(u.tick)


trio.run(main)
