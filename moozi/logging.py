import os
import time
from typing import Any, NamedTuple, Dict
import uuid
from pathlib import Path
from acme.utils.loggers.terminal import TerminalLogger

import haiku as hk
import jax.numpy as jnp
from absl import logging

import moozi as mz


class JAXBoardStepData(NamedTuple):
    scalars: Dict[str, Any]
    histograms: Dict[str, Any]

    def update(self, other: "JAXBoardStepData"):
        self.scalars.update(other.scalars)
        self.histograms.update(other.histograms)

    def add_hk_params(self, params):
        for module_name, weight_name, weights in hk.data_structures.traverse(params):
            name = module_name + "/" + weight_name
            assert name not in self.histograms
            self.histograms[name] = weights


class Data(NamedTuple):
    # TODO: replace JAXBoardStepData with this class
    name: str
    content: jnp.ndarray


def is_scalar(datum: Data):
    return jnp.isscalar(datum.content)


def is_vector(datum: Data):
    return jnp.ndim(datum.content) == 1


class Logger:
    pass


class FileLogger(Logger):
    # TODO: implement this
    def __init__(self, name, fname, time_delta: float = 0.0):
        self._name = name
        curr_dir_path = Path(os.path.abspath("."))
        self._f = curr_dir_path.open("w")
        self._time_delta = time_delta
        self._time = time.time()
        self._steps = 0
        print(f"{self._name} is logging to {curr_dir_path}")

    def write(self, data: Data):
        now = time.time()
        if (now - self._time) > self._time_delta:
            self._write_now(data)
            self._time = now
        self._steps += 1

    def _write_now(self, data: Data):
        if is_scalar(data):
            self._f.write(f"{str(self._steps).ljust(3)} {data.name} {data.content}\n")
        # elif is_vector(data):
        #     np.histogram(data.content, bins=10, range=(0, 1))
        #     self._f.write(f"{str(self._steps).ljust(3)} {data.name} {data.content.shape}\n")

    def close(self):
        r"""
        Always call this method the logging is finished.
        Otherwise unexpected hangings may occur.
        """
        return self._writer.close()


class JAXBoardLogger(Logger):
    def __init__(self, name, log_dir=None, time_delta: float = 0.0):
        self._name = name
        self._log_dir = log_dir or "./tensorboard_log/"
        self._log_dir = str(Path(self._log_dir).resolve())
        self._time_delta = time_delta
        self._time = time.time()
        self._steps = 0
        self._writer = mz.jaxboard.SummaryWriter(name, log_dir=self._log_dir)
        logging.info(f"{self._name} is logging to {(self._log_dir)}")

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
