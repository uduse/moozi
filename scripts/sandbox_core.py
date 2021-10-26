from acme.utils import loggers
import cloudpickle
import operator
import os
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    List
)

import acme
import attr
import chex
import dm_env
import jax
import jax.numpy as jnp
import moozi as mz
import numpy as np
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

