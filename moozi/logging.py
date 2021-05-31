import os
import time
import typing
import uuid

import acme
import haiku as hk
import jax.numpy as jnp
import jaxboard

import moozi as mz


def get_log_dir():
    guild_ai_run_dir_key = "RUN_DIR"
    env_items = dict(os.environ.items())
    if guild_ai_run_dir_key in env_items:
        log_dir = env_items[guild_ai_run_dir_key]
    else:
        log_dir = "/tmp/moozi-log-" + str(uuid.uuid4())[:16]
    return log_dir


class JAXBoardStepData(typing.NamedTuple):
    scalars: typing.Dict[str, jnp.DeviceArray]
    histograms: typing.Dict[str, jnp.DeviceArray]

    def update(self, other: "JAXBoardStepData"):
        self.scalars.update(other.scalars)
        self.histograms.update(other.histograms)

    def add_hk_params(self, params):
        for module_name, weight_name, weights in hk.data_structures.traverse(params):
            name = module_name + "/" + weight_name
            assert name not in self.histograms
            self.histograms[name] = weights


class JAXBoardLogger(acme.utils.loggers.base.Logger):
    def __init__(self):
        self._time = time.time()
        self._steps = 0
        self._log_dir = get_log_dir()
        self._writer = jaxboard.SummaryWriter(log_dir=self._log_dir)

    def write(self, data: JAXBoardStepData):
        for key in data.scalars:
            self._writer.scalar(key, data.scalars[key], step=self._steps)
        for key in data.histograms:
            num_bins = 50
            self._writer.histogram(
                key, data.histograms[key], num_bins, step=self._steps
            )
        self._steps += 1
