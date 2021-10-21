import collections
from acme.utils import loggers
import cloudpickle
import operator
import os
from dataclasses import InitVar, dataclass, field
from functools import partial
import random
from typing import (
    Any,
    Callable,
    Coroutine,
    Deque,
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

import acme
import attr
import chex
import dm_env
import jax
import jax.numpy as jnp
import moozi as mz
import numpy as np
import open_spiel
import optax
import ray
import rlax
import tree
import trio
import trio_asyncio
from absl import logging
from acme.utils.tree_utils import stack_sequence_fields, unstack_sequence_fields
from acme.wrappers import SinglePrecisionWrapper
from jax._src.numpy.lax_numpy import stack
from moozi.batching_layer import BatchingClient, BatchingLayer
from moozi.learner import TrainingState
from moozi.link import UniverseAsync, link
from moozi.logging import JAXBoardStepData
from moozi.nn import NeuralNetwork
from moozi.replay import StepSample, TrajectorySample
from moozi.utils import SimpleBuffer
from trio_asyncio import aio_as_trio
from acme.utils.loggers import TerminalLogger

from moozi.replay import make_target_from_traj
from acme_openspiel_wrapper import OpenSpielWrapper
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
    timestep: dm_env.TimeStep = None
    obs: np.ndarray = None
    is_first: bool = True
    is_last: bool = False
    to_play: int = -1
    reward: float = 0.0
    action: int = -1
    discount: float = 1.0
    legal_actions_mask: np.ndarray = None

    # planner
    # root: Any = None
    root_value: float = 0
    action_probs: Any = None

    # player
    stacked_frames: np.ndarray = None

    output_buffer: tuple = field(default_factory=tuple)


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


# @link
# def set_legal_actions(timestep):
#     # TODO: make a part of the environment law
#     return dict(legal_actions_mask=get_legal_actions(timestep))


@link
def update_episode_stats(timestep, sum_episodic_reward, num_episodes, universe_id):
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
def output_reward(is_last, reward, output_buffer):
    if is_last:
        output_buffer = output_buffer + (reward,)
        return dict(output_buffer=output_buffer)


@ray.remote
@dataclass
class LoggerActor:
    logger: mz.logging.Logger

    def log(self, data):
        self.logger.write(data)


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
        if timestep.reward is None:
            reward = 0.0
        else:
            reward = float(np.nan_to_num(timestep.reward))
        return dict(
            timestep=timestep,
            obs=get_observation(timestep),
            is_first=timestep.first(),
            is_last=timestep.last(),
            to_play=self.env_state.current_player,
            reward=reward,
            legal_actions_mask=get_legal_actions(timestep),
        )


@dataclass(repr=False)
class RolloutWorker:
    batching_layers: Optional[List[BatchingLayer]] = None
    universes: Optional[List[UniverseAsync]] = None

    def setup(
        self, factory: Callable[[], Tuple[List[BatchingLayer], List[UniverseAsync]]]
    ):
        self.batching_layers, self.universes = factory()

    def set_verbosity(self, verbosity):
        logging.set_verbosity(verbosity)

    def run(self, num_ticks):
        async def main_loop():
            async with trio.open_nursery() as main_nursery:
                for b in self.batching_layers:
                    b.is_paused = False
                    main_nursery.start_soon(b.start_processing)
                    # TODO: toggle logging
                    # main_nursery.start_soon(b.start_logging)

                async with trio.open_nursery() as universe_nursery:
                    for u in self.universes:
                        universe_nursery.start_soon(partial(u.tick, times=num_ticks))

                for b in self.batching_layers:
                    b.is_paused = True

        trio_asyncio.run(main_loop)

        return self.flush_output_buffers()

    def flush_output_buffers(self) -> List[TrajectorySample]:
        outputs: List[TrajectorySample] = sum(
            (list(u.artifact.output_buffer) for u in self.universes), []
        )

        for u in self.universes:
            u.artifact.output_buffer = tuple()
        return outputs


@link
def set_random_action_from_timestep(timestep: dm_env.TimeStep):
    action = -1
    if not timestep.last():
        legal_actions = timestep.observation[0].legal_actions
        random_action = np.random.choice(np.flatnonzero(legal_actions == 1))
        action = random_action
    return dict(action=action)


def make_sgd_step_fn(network: mz.nn.NeuralNetwork, loss_fn: mz.loss.LossFn, optimizer):
    @jax.jit
    @chex.assert_max_traces(n=1)
    def sgd_step_fn(training_state: TrainingState, batch: mz.replay.TrainTarget):
        # gradient descend
        _, new_key = jax.random.split(training_state.rng_key)
        grads, extra = jax.grad(loss_fn, has_aux=True, argnums=1)(
            network, training_state.params, batch
        )
        updates, new_opt_state = optimizer.update(grads, training_state.opt_state)
        new_params = optax.apply_updates(training_state.params, updates)
        steps = training_state.steps + 1
        new_training_state = TrainingState(new_params, new_opt_state, steps, new_key)

        # KL calculation
        orig_logits = network.initial_inference(
            training_state.params, batch.stacked_frames
        ).policy_logits
        new_logits = network.initial_inference(
            new_params, batch.stacked_frames
        ).policy_logits
        prior_kl = jnp.mean(rlax.categorical_kl_divergence(orig_logits, new_logits))

        # store data to log
        step_data = mz.logging.JAXBoardStepData(scalars={}, histograms={})
        step_data.update(extra)
        step_data.scalars["prior_kl"] = prior_kl
        # step_data.histograms["reward"] = batch.last_reward
        step_data.add_hk_params(new_params)

        return new_training_state, step_data

    return sgd_step_fn


class ParameterOptimizer:
    def __init__(self, network, params, loss_fn, optimizer, loggers=None):
        self.state = TrainingState(
            params=params,
            opt_state=optimizer.init(params),
            steps=0,
            rng_key=jax.random.PRNGKey(0),
        )
        self.sgd_step_fn = make_sgd_step_fn(network, loss_fn, optimizer)
        self.loggers = [] if not loggers else loggers
        self._num_updates: int = 0

    def update(self, batch):
        if len(batch) == 0:
            raise ValueError("Batch is empty")
        self.state, step_data = self.sgd_step_fn(self.state, batch)
        self._num_updates += 1
        self._log_step_data(step_data)
        return self.state.params

    def get_params(self):
        return self.state.params

    def save(self, path):
        with open(path, "wb") as f:
            cloudpickle.dump(self, f)

    @classmethod
    def restore(path):
        with open(path, "rb") as f:
            return cloudpickle.load(f)

    def set_loggers(self, loggers_factory: Callable[[], List[mz.logging.Logger]]):
        self.loggers = loggers_factory()
        logging.info("setting loggers")

    def _log_step_data(self, step_data):
        for logger in self.loggers:
            if isinstance(logger, TerminalLogger):
                logger.write(step_data.scalars)
            elif isinstance(logger, mz.logging.JAXBoardLogger):
                logger.write(step_data)
            else:
                raise NotImplementedError(f"Logger type {type(logger)} not supported")

    def get_stats(self):
        return dict(num_updates=self._num_updates)

    def log_stats(self):
        logging.info(self.get_stats())

    def close(self):
        for logger in self._loggers:
            if isinstance(logger, mz.logging.JAXBoardLogger):
                logging.info(logger._name, "closed")
                logger.close()


@dataclass(repr=False)
class InferenceServer:
    network: mz.nn.NeuralNetwork = field(init=False)
    params: Any = field(init=False)

    loggers: List[mz.logging.Logger] = field(default_factory=list)

    _num_set_params: int = 0
    _num_set_network: int = 0

    _num_init_inf: int = 0
    _init_inf_timer = mz.utils.WallTimer()
    # _num_init_inf_avg_size: float = 0.0
    _num_recurr_inf: int = 0

    def set_network(self, network):
        logging.info("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
        logging.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
        logging.info(f"jax.devices(): {jax.devices()}")
        self.network = network
        self.init_inf_fn = jax.jit(network.initial_inference)
        self.recurr_inf_fn = jax.jit(network.recurrent_inference)
        self._num_set_network += 1

    def set_params(self, params):
        self.params = params
        self._num_set_params += 1

    def set_loggers(self, loggers_factory: Callable[[], List[mz.logging.Logger]]):
        self.loggers = loggers_factory()
        logging.info("setting loggers")

    def init_inf(self, list_of_stacked_frames):
        batch_size = len(list_of_stacked_frames)
        results = self.init_inf_fn(self.params, np.array(list_of_stacked_frames))
        results = tree.map_structure(np.array, results)
        self._num_init_inf += 1
        return unstack_sequence_fields(results, batch_size)

    def recurr_inf(self, inputs):
        batch_size = len(inputs)
        hidden_states = np.array(list(map(operator.itemgetter(0), inputs)))
        actions = np.array(list(map(operator.itemgetter(1), inputs)))
        results = self.recurr_inf_fn(self.params, hidden_states, actions)
        results = tree.map_structure(np.array, results)
        self._num_recurr_inf += 1
        return unstack_sequence_fields(results, batch_size)

    def get_stats(self):
        return dict(
            num_set_network=self._num_set_network,
            num_set_params=self._num_set_params,
            num_init_inf=self._num_init_inf,
            num_recurr_infs=self._num_recurr_inf,
        )

    def log_stats(self):
        logging.info(self.get_stats())

    @staticmethod
    def make_init_inf_remote_fn(handle):
        """For using with Ray."""

        async def init_inf_remote(x):
            return await aio_as_trio(handle.init_inf.remote(x))

        return init_inf_remote

    @staticmethod
    def make_recurr_inf_remote_fn(handle):
        """For using with Ray."""

        async def recurr_inf_remote(x):
            return await aio_as_trio(handle.recurr_inf.remote(x))

        return recurr_inf_remote


@link
@dataclass
class StepSampleSaver:
    traj_buffer: list = field(default_factory=list)

    def __call__(
        self,
        obs,
        action,
        reward,
        root_value,
        is_first,
        is_last,
        action_probs,
        output_buffer,
    ):
        step_record = StepSample(
            frame=obs,
            reward=reward,
            is_first=is_first,
            is_last=is_last,
            action=action,
            root_value=root_value,
            action_probs=action_probs,
        ).cast()

        self.traj_buffer.append(step_record)

        if is_last:
            traj = stack_sequence_fields(self.traj_buffer)
            self.traj_buffer.clear()
            return dict(output_buffer=output_buffer + (traj,))


@dataclass(repr=False)
class ReplayBuffer:
    # TODO: remove config here
    config: Config

    store: Deque[TrajectorySample] = field(init=False)

    def __post_init__(self):
        self.store = collections.deque(maxlen=self.config.replay_buffer_size)

    def add_samples(self, samples: List[TrajectorySample]):
        self.store.extend(samples)
        logging.info(f"Replay buffer size: {self.size()}")
        return self.size()

    def get_batch(self, batch_size=1):
        if not self.store:
            raise ValueError("Empty replay buffer")

        trajs = random.choices(self.store, k=batch_size)
        batch = []
        for traj in trajs:
            random_start_idx = random.randrange(len(traj.reward))
            target = make_target_from_traj(
                traj,
                start_idx=random_start_idx,
                discount=1.0,
                num_unroll_steps=self.config.num_unroll_steps,
                num_td_steps=self.config.num_td_steps,
                num_stacked_frames=self.config.num_stacked_frames,
            )
            batch.append(target)
        return stack_sequence_fields(batch)

    # def add_and_get_batch(self, samples: List[TrajectorySample], batch_size=1):
    #     self.add_samples(samples)
    #     return self.get_batch(batch_size=batch_size)

    def size(self):
        return len(self.store)


class MetricsReporter:
    def __init__(self) -> None:
        self.logger = mz.logging.JAXBoardLogger(name="reporter")

    def report(self, step_data: JAXBoardStepData):
        self.logger.write(step_data)


MetricsReporterActor = ray.remote(num_cpus=0)(MetricsReporter)
