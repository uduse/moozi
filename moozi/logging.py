import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, NamedTuple

import haiku as hk
import jax.numpy as jnp
import numpy as np
import ray
from absl import logging
from acme.utils.loggers.terminal import TerminalLogger

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


class Logger:
    pass


class StepDataV2:
    pass


@dataclass
class StepDataText(StepDataV2):
    text: str


@dataclass
class StepDataImage(StepDataV2):
    image: np.ndarray


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


class JAXBoardLoggerV2(Logger):
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


class MetricsReporter:
    def __init__(self) -> None:
        self.loggers = [mz.logging.JAXBoardLogger(name="reporter")]

    def report(self, step_data: JAXBoardStepData):
        for logger in self.loggers:
            logger.write(step_data)


MetricsReporterActor = ray.remote(num_cpus=0)(MetricsReporter)
