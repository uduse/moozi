import collections
import functools
from typing import Any, Callable, Coroutine, Dict, List, NamedTuple, Optional, Union

import acme
import attr
import dm_env
import moozi as mz
import numpy as np
import open_spiel
import ray
import trio
import trio_asyncio
from absl import logging
from acme.wrappers import SinglePrecisionWrapper
from acme.wrappers.open_spiel_wrapper import OpenSpielWrapper
from jax._src.numpy.lax_numpy import stack
from moozi.batching_layer import BatchingClient, BatchingLayer
from moozi.link import AsyncUniverse, link
from moozi.utils import SimpleBuffer
from trio_asyncio import aio_as_trio


def make_catch():
    env_columns, env_rows = 3, 3
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


def get_observation(timestep: dm_env.TimeStep):
    if isinstance(timestep.observation, list):
        assert len(timestep.observation) == 1
        return timestep.observation[0].observation
    else:
        raise NotImplementedError


@attr.s(auto_attribs=True, repr=False)
class Artifact:
    # meta
    universe_id: int = -1
    num_ticks: int = 0
    num_episodes: int = 0
    avg_episodic_reward: float = 0
    sum_episodic_reward: float = 0

    # environment
    env_state: dm_env.Environment = None
    timestep: dm_env.TimeStep = None
    to_play: int = -1
    action: int = -1

    # player
    stacked_frames: np.ndarray = None


@link
@attr.s(repr=False)
class FrameStacker:
    num_frames: int = attr.ib(default=1)

    _padding: Optional[np.ndarray] = attr.ib(default=None)
    _deque: collections.deque = attr.ib(init=False)

    def __attrs_post_init__(self):
        self._deque = collections.deque(maxlen=self.num_frames)

    def __call__(self, timestep) -> Any:
        if self._padding is None:
            self._make_padding(timestep)

        if timestep.last():
            self._deque.clear()

        self._deque.append(get_observation(timestep))

        stacked_frames = self._get_stacked_frames()

        return dict(stacked_frames=stacked_frames)

    def _get_stacked_frames(self):
        stacked_frames = np.array(list(self._deque))
        num_frames_to_pad = self.num_frames - len(self._deque)
        if num_frames_to_pad > 0:
            paddings = np.stack(
                [np.copy(self._padding) for _ in range(num_frames_to_pad)], axis=0
            )
            stacked_frames = np.append(paddings, np.array(list(self._deque)), axis=0)
        return stacked_frames

    def _make_padding(self, timestep):
        assert timestep.first()
        shape = get_observation(timestep).shape
        self._padding = np.zeros(shape)


@link
def wrap_up_episode(timestep, sum_episodic_reward, num_episodes, universe_id):
    if timestep.last():
        sum_episodic_reward = sum_episodic_reward + float(timestep.reward)
        num_episodes = num_episodes + 1
        avg_episodic_reward = round(sum_episodic_reward / num_episodes, 3)

        result = dict(
            num_episodes=num_episodes,
            sum_episodic_reward=sum_episodic_reward,
            avg_episodic_reward=avg_episodic_reward,
        )
        logging.debug({**dict(universe_id=universe_id), **result})

        return result


@link
def increment_tick(num_ticks):
    return {"num_ticks": num_ticks + 1}


@link
@attr.s(repr=False)
class PlayerShell:
    client: BatchingClient = attr.ib()

    async def __call__(self, timestep):
        await trio.sleep(0)
        if not timestep.last():
            await self.client.send_request(timestep)
            result = await self.client.receive_response()
            logging.debug(f"{self.client.client_id} got {result}")
            return result


@link
@attr.s(repr=False)
class EnvironmentLaw:
    env_state = attr.ib()

    def __call__(self, timestep: dm_env.TimeStep, action: int):
        if timestep is None or timestep.last():
            timestep = self.env_state.reset()
        else:
            timestep = self.env_state.step([action])
        return {"timestep": timestep}


@attr.s
class InteractionManager:
    batching_layers = attr.ib()

    universes = attr.ib(init=False)

    num_universes: int = attr.ib(default=2)
    num_ticks: int = attr.ib(default=10)
    verbosity = attr.ib(default=logging.INFO)

    def __attrs_post_init__(self):
        logging.set_verbosity(self.verbosity)

    def setup(self):
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

    def run(self):
        self.setup()

        import trio

        async def main_loop():
            async with trio.open_nursery() as main_nursery:
                for b in self.batching_layers:
                    main_nursery.start_soon(b.start_processing)

                async with trio.open_nursery() as universe_nursery:
                    for u in self.universes:
                        universe_nursery.start_soon(
                            functools.partial(u.tick, times=self.num_ticks)
                        )

                for b in self.batching_layers:
                    await b.close()

        trio_asyncio.run(main_loop)

        return self.universes[0].artifact


def builder():
    batching_layer = BatchingLayer(
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

    policy_server = ray.remote(PolicyServer).remote()
