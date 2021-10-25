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


@ray.remote
@dataclass
class LoggerActor:
    logger: mz.logging.Logger

    def log(self, data):
        self.logger.write(data)


def make_sgd_step_fn(network: mz.nn.NeuralNetwork, loss_fn: mz.loss.LossFn, optimizer):
    # @partial(jax.jit, backend="cpu")
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


@dataclass(repr=False)
class ParameterOptimizer:
    network: mz.nn.NeuralNetwork = field(init=False)
    state: TrainingState = field(init=False)
    sgd_step_fn: Callable = field(init=False)

    loggers: List[mz.logging.Logger] = field(default_factory=list)

    _num_updates: int = 0

    def build(self, factory):
        network, params, loss_fn, optimizer = factory()
        self.network = network
        self.state = TrainingState(
            params=params,
            opt_state=optimizer.init(params),
            steps=0,
            rng_key=jax.random.PRNGKey(0),
        )
        self.sgd_step_fn = make_sgd_step_fn(network, loss_fn, optimizer)

    def update(self, batch):
        if len(batch) == 0:
            raise ValueError("Batch is empty")
        self.state, step_data = self.sgd_step_fn(self.state, batch)
        self._num_updates += 1
        self._log_step_data(step_data)

    def get_params(self):
        return self.state.params

    def get_network(self):
        return self.network

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
class TrajectoryOutputWriter:
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


class MetricsReporter:
    def __init__(self) -> None:
        self.loggers = [mz.logging.JAXBoardLogger(name="reporter")]

    def report(self, step_data: JAXBoardStepData):
        for logger in self.loggers:
            logger.write(step_data)


MetricsReporterActor = ray.remote(num_cpus=0)(MetricsReporter)
