import collections
from dataclasses import InitVar, dataclass, field
import functools
import operator
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
from acme.utils.tree_utils import unstack_sequence_fields

import jax
import jax.numpy as jnp
import acme
import attr
import dm_env
import tree
import moozi as mz
import numpy as np
import open_spiel
import ray
import trio
import trio_asyncio
from absl import logging
from acme.wrappers import SinglePrecisionWrapper
from acme_openspiel_wrapper import OpenSpielWrapper
from jax._src.numpy.lax_numpy import stack
from moozi.batching_layer import BatchingClient, BatchingLayer
from moozi.link import UniverseAsync, link
from moozi.nn import NeuralNetwork
from moozi.utils import SimpleBuffer
from trio_asyncio import aio_as_trio
from config import Config


def make_catch():
    prev_verbosity = logging.get_verbosity()
    logging.set_verbosity(logging.WARNING)

    env_columns, env_rows = 6, 6
    raw_env = open_spiel.python.rl_environment.Environment(
        f"catch(columns={env_columns},rows={env_rows})"
    )
    env = OpenSpielWrapper(raw_env)
    env = SinglePrecisionWrapper(env)
    env_spec = acme.specs.make_environment_spec(env)

    logging.set_verbosity(prev_verbosity)
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


def get_legal_actions(timestep: dm_env.TimeStep):
    if isinstance(timestep.observation, list):
        assert len(timestep.observation) == 1
        return timestep.observation[0].legal_actions
    else:
        raise NotImplementedError


@dataclass
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
    legal_actions_mask: np.ndarray = None
    stacked_frames: np.ndarray = None


@link
@dataclass
class FrameStacker:
    num_frames: int = 1

    padding: Optional[np.ndarray] = None
    deque: collections.deque = None

    def __post_init__(self):
        self.deque = collections.deque(maxlen=self.num_frames)

    def __call__(self, timestep) -> Any:
        if self.padding is None:
            self._make_padding(timestep)

        if timestep.last():
            self.deque.clear()

        self.deque.append(get_observation(timestep))

        stacked_frames = self._get_stacked_frames()

        return dict(stacked_frames=stacked_frames)

    def _get_stacked_frames(self):
        stacked_frames = np.array(list(self.deque))
        num_frames_to_pad = self.num_frames - len(self.deque)
        if num_frames_to_pad > 0:
            paddings = np.stack(
                [np.copy(self.padding) for _ in range(num_frames_to_pad)], axis=0
            )
            stacked_frames = np.append(paddings, np.array(list(self.deque)), axis=0)
        return stacked_frames

    def _make_padding(self, timestep):
        assert timestep.first()
        shape = get_observation(timestep).shape
        self.padding = np.zeros(shape)


@link
def set_legal_actions(timestep):
    return dict(legal_actions_mask=get_legal_actions(timestep))


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
@dataclass
class EnvironmentLaw:
    env_state: dm_env.Environment

    def __call__(self, timestep: dm_env.TimeStep, action: int):
        if timestep is None or timestep.last():
            timestep = self.env_state.reset()
        else:
            timestep = self.env_state.step([action])
        return dict(timestep=timestep)


@dataclass(repr=False)
class InteractionManager:
    batching_layers: Optional[List[BatchingLayer]] = None
    universes: Optional[List[UniverseAsync]] = None

    def setup(
        self,
        factory: Callable[[], Tuple[List[BatchingLayer], List[UniverseAsync]]],
    ):
        self.batching_layers, self.universes = factory()

    def set_verbosity(self, verbosity):
        logging.set_verbosity(verbosity)

    def run(self, num_ticks):

        import trio

        async def main_loop():
            async with trio.open_nursery() as main_nursery:
                for b in self.batching_layers:
                    main_nursery.start_soon(b.start_processing)
                    main_nursery.start_soon(b.start_logging)

                async with trio.open_nursery() as universe_nursery:
                    for u in self.universes:
                        universe_nursery.start_soon(
                            functools.partial(u.tick, times=num_ticks)
                        )

                for b in self.batching_layers:
                    await b.close()

        trio_asyncio.run(main_loop)
        return [u.artifact for u in self.universes]


@dataclass(repr=False)
class BatchInferenceServer:
    network: InitVar
    params: InitVar

    init_inf_fn: Callable = field(init=False)
    recurr_inf_fn: Callable = field(init=False)

    def __post_init__(self, network, params):
        import os
        import jax

        logging.info("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
        logging.info(
            "CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"])
        )
        logging.info(f"seeing {jax.devices()}")
        self.init_inf_fn = functools.partial(jax.jit(network.initial_inference), params)
        self.recurr_inf_fn = functools.partial(
            jax.jit(network.recurrent_inference), params
        )
        logging.info(f"seeing {jax.devices()}")

    def init_inf(self, frames):
        batch_size = len(frames)
        results = self.init_inf_fn(np.array(frames))
        results = tree.map_structure(np.array, results)
        return unstack_sequence_fields(results, batch_size)

    def recurr_inf(self, inputs):
        batch_size = len(inputs)
        hidden_states = np.array(list(map(operator.itemgetter(0), inputs)))
        actions = np.array(list(map(operator.itemgetter(1), inputs)))
        results = self.recurr_inf_fn(hidden_states, actions)
        results = tree.map_structure(np.array, results)
        return unstack_sequence_fields(results, batch_size)


def setup_config(config: Config):
    def make_artifact(index):
        return Artifact(universe_id=index)

    config.set(Config.ARTIFACT_FACTORY, make_artifact)

    config.set(Config.ENV_FACTORY, lambda: make_catch()[0])
    config.set(Config.ENV_SPEC, make_catch()[1])
    config.set(Config.NUM_UNIVERSES, 2000)
