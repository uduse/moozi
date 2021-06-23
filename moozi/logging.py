import os
import time
import typing
import uuid
from pathlib import Path

import acme
import haiku as hk
import jax.numpy as jnp

import moozi as mz


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
    def __init__(self, name, log_dir=None, time_delta: float = 0.0):
        self._name = name
        self._log_dir = log_dir or "./tensorboard_log/"
        self._log_dir = str(Path(self._log_dir).resolve())
        self._time_delta = time_delta
        self._time = time.time()
        self._steps = 0
        self._writer = mz.jaxboard.SummaryWriter(name, log_dir=self._log_dir)
        print(f"{self._name} is logging to {(self._log_dir)}")

    def write(self, data: JAXBoardStepData):
        now = time.time()
        if (now - self._time) > self._time_delta:
            self._write_now(data)
            self._time = now
        self._steps += 1

    def _write_now(self, data: JAXBoardStepData):
        for key in data.scalars:
            prefixed_key = self._name + ":" + key
            self._writer.scalar(
                tag=prefixed_key,
                value=data.scalars[key],
                step=self._steps,
            )
        # self._writer.scalar("time", time.time())
        for key in data.histograms:
            prefixed_key = self._name + ":" + key
            num_bins = 50
            self._writer.histogram(
                tag=prefixed_key,
                values=data.histograms[key],
                bins=num_bins,
                step=self._steps,
            )

    def close(self):
        r"""
        Always call this method the logging is finished.
        Otherwise unexpected hangings may occur.
        """
        return self._writer.close()
