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
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Union

import trio
import acme
import dm_env
import jax
from matplotlib import artist
import moozi as mz
import numpy as np
import open_spiel
import ray
import tree
from acme.agents.jax.impala.types import Observation
from acme.jax.networks.base import Value
from acme.wrappers import SinglePrecisionWrapper
from acme.wrappers.open_spiel_wrapper import OpenSpielWrapper
from moozi.utils import SimpleQueue

# %%
ray.init(ignore_reinit_error=True)

# %%
# def env_factory():
#     env_columns, env_rows = 5, 5
#     raw_env = open_spiel.python.rl_environment.Environment(
#         f"catch(columns={env_columns},rows={env_rows})"
#     )
#     env = OpenSpielWrapper(raw_env)
#     env = SinglePrecisionWrapper(env)
#     env_spec = acme.specs.make_environment_spec(env)
#     return env, env_spec


def env_factory():
    raw_env = open_spiel.python.rl_environment.Environment(f"tic_tac_toe")
    env = OpenSpielWrapper(raw_env)
    env = SinglePrecisionWrapper(env)
    env_spec = acme.specs.make_environment_spec(env)
    return env, env_spec


# %%

# %%
@dataclasses.dataclass
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


class _Link:
    def __init__(
        self,
        callable_obj: Callable[..., Optional[Dict[str, Any]]],
        to_read: Union[List[str], str] = "auto",
        to_write: Union[List[str], str] = "auto",
    ) -> None:
        self._callable_obj = callable_obj
        self._to_read = to_read
        self._to_write = to_write

    def __call__(self, artifact: Artifact):
        keys_to_read = self._get_keys_to_read(artifact)
        artifact_window = _Link._read_artifact(artifact, keys_to_read)
        updates = self._callable_obj(**artifact_window)
        if not updates:
            updates = {}
        self._validate_updates(artifact, updates)
        _Link._update_artifact(artifact, updates)
        return updates

    def __getattribute__(self, name: str) -> Any:
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return getattr(self._callable_obj, name)

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
        if not _Link._artifact_has_keys(artifact, list(updates.keys())):
            raise ValueError

        if self._to_write == "auto":
            pass
        elif isinstance(self._to_write, list):
            update_nothing = (not self._to_write) and (not updates)
            if update_nothing:
                pass
            elif self._to_write != list(updates.keys()):
                raise ValueError("write_view keys mismatch.")
        else:
            raise ValueError("`to_write` type not accepted.")

    def _wrapped_func_keys(self):
        return set(inspect.signature(self._callable_obj).parameters.keys())

    def _get_keys_to_read(self, artifact):
        if self._to_read == "auto":
            keys = self._wrapped_func_keys()
            keys = keys - {"self"}  ## TODO?
            if not _Link._artifact_has_keys(artifact, keys):
                raise ValueError(f"{str(keys)} not in {str(artifact.__dict__.keys())})")
        elif isinstance(self._to_read, list):
            keys = self._to_read
        else:
            raise ValueError("`to_read` type not accepted.")
        return keys


class _LinkClassWrapper:
    def __init__(self, class_) -> None:
        self._class = class_

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return link(self._class(*args, **kwargs))


def link(*args, **kwargs):
    if len(args) == 1 and not kwargs and inspect.isclass(args[0]):
        return _LinkClassWrapper(args[0])
    elif len(args) == 1 and not kwargs and callable(args[0]):
        func = args[0]
        return _Link(func, to_read="auto", to_write="auto")
    else:
        func = functools.partial(_Link, *args, **kwargs)
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


def tick(artifact, linked_laws):
    for law in linked_laws:
        law(artifact)


@link
def environment_law(
    env_state: dm_env.Environment, timestep: dm_env.TimeStep, action: int
):
    if timestep is None or timestep.last():
        timestep = env_state.reset()
    else:
        timestep = env_state.step([action])
    return {"env_state": env_state, "timestep": timestep}


@link
def increment_tick(num_ticks):
    return {"num_ticks": num_ticks + 1}


class UniverseWorker:
    def __init__(self, artifact_factory, laws_factory):
        self._artifact = artifact_factory()
        self._laws = laws_factory()

    def tick(self):
        tick(self._artifact, self._laws)

    def loop(self):
        while True:
            time.sleep(1)
            self.tick()

    def artifact(self):
        return self._artifact

    def laws(self):
        return self._laws


# %%
def artifact_factory():
    artifact = Artifact()
    artifact.env_state = env_factory()[0]
    return artifact


def laws_factory():
    return [
        environment_law,
        random_action_law,
        link(lambda timestep: print("reward:", timestep.reward)),
        increment_tick,
    ]


# %%
@link
class C:
    def __init__(self, index) -> None:
        self._index = index

    def __call__(self, num_ticks) -> Any:
        print(f"{self._index} : {num_ticks}")


# %%
worker = UniverseWorker(artifact_factory, lambda: [C(0), C(1), increment_tick])
worker.tick()
worker.tick()
worker.tick()
worker.tick()


# %%
c = C(100)

# %%
a = artifact_factory()
c(a)

# %%
# class BatchingLayer:
#     def __init__(self) -> None:
#         self._q = []

#     async def add(self, obs):
#         pass

#     async def run(self):
#         self

num_simulators = 10
batch_size = 9

async def batching_layer(receive_request_channel, send_responses_channels):
    print("batch processor starting")
    async with contextlib.AsyncExitStack() as stack:
        for context in [receive_request_channel, *send_responses_channels]:
            await stack.enter_async_context(context)

        batch = []
        tick_timeframe = 1
        # last_tick_time = time.time()
        while True:
            try:
                mode = 'move on'
                with trio.move_on_after(tick_timeframe):
                    while True:
                        data = await receive_request_channel.receive()
                        batch.append(data)
                        if len(batch) >= batch_size:
                            mode = 'skipped'
                            break

                print(mode)
                await process_batch(send_responses_channels, batch)
                batch.clear()

            except trio.EndOfChannel:
                assert len(batch) == 0
                break

    print("batch processor closed")


async def process_batch(send_responses_channels, batch):
    results = [(idx, c + 100) for idx, c in batch]
    print(f"processed {len(batch)} tasks")

    for idx, data in results:
        await send_responses_channels[idx].send((idx, data))


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
    async with trio.open_nursery() as nursery:
        send_request_channel, receive_request_channel = trio.open_memory_channel(0)

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


trio.run(main)
