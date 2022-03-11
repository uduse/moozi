import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Union

import haiku as hk
import jax.numpy as jnp
import numpy as np
import ray
from absl import logging

import moozi as mz


@dataclass
class LogDatum:
    tag: str

    @staticmethod
    def from_dict(d: dict) -> List["LogDatum"]:
        assert isinstance(d, dict)
        mappers = (
            (int, LogScalar),
            (float, LogScalar),
            (str, LogText),
            (np.array, LogHistogram),
        )

        def process(key, val):
            if isinstance(val, jnp.ndarray):
                if val.size == 1:
                    return LogScalar(key, float(val))
                else:
                    return LogHistogram(key, np.array(val))

            for cast_fn, target_cls in mappers:
                try:
                    val = cast_fn(val)
                except (ValueError, TypeError):
                    pass
                else:
                    return target_cls(key, val)

            raise ValueError(f"Unable to process {key}={val}, type={type(val)}")

        return [process(k, v) for k, v in d.items()]

    @staticmethod
    def from_any(data: Union[List["LogDatum"], "LogDatum", dict]) -> List["LogDatum"]:
        if isinstance(data, LogDatum):
            return [data]
        elif isinstance(data, dict):
            return LogDatum.from_dict(data)
        else:
            return data


@dataclass
class LogScalar(LogDatum):
    scalar: float


@dataclass
class LogText(LogDatum):
    text: str


@dataclass
class LogImage(LogDatum):
    image: np.ndarray


@dataclass
class LogHistogram(LogDatum):
    values: np.ndarray


class Logger:
    def write(self, data: Union[List[LogDatum], LogDatum, dict]):
        raise NotImplementedError


class JAXBoardLoggerV2(Logger):
    def __init__(self, name="logger", log_dir=None, time_delta: float = 0.0):
        self._name = name
        self._log_dir = log_dir or "./tensorboard_log/"
        self._log_dir = str(Path(self._log_dir).resolve())
        self._time_delta = time_delta
        self._time = time.time()
        self._steps = 0
        self._writer = mz.jaxboard.SummaryWriter(name, log_dir=self._log_dir)
        logging.info(f"{self._name} is logging to {(self._log_dir)}")

    def write(self, data: Union[List[LogDatum], LogDatum, dict]):
        data_ready = LogDatum.from_any(data)

        now = time.time()
        if (now - self._time) > self._time_delta:
            for datum in data_ready:
                self._write_now_datum(datum)
            self._time = now

        self._steps += 1

    def set_steps(self, steps):
        self._steps = steps

    def _write_now_datum(self, datum: LogDatum):
        prefixed_key = self._name + ":" + datum.tag
        if isinstance(datum, LogText):
            self._writer.text(tag=prefixed_key, textdata=datum.text, step=self._steps)
        elif isinstance(datum, LogImage):
            self._writer.image(tag=prefixed_key, image=datum.image, step=self._steps)
        elif isinstance(datum, LogScalar):
            self._writer.scalar(tag=prefixed_key, value=datum.scalar, step=self._steps)
        elif isinstance(datum, LogHistogram):
            num_bins = 50
            self._writer.histogram(
                tag=prefixed_key,
                values=datum.values,
                bins=num_bins,
                step=self._steps,
            )
        else:
            raise ValueError(f"Unsupported datum type: {type(datum)}")

    def close(self):
        r"""
        Always call this method the logging is finished.
        Otherwise unexpected hangings may occur.
        """
        return self._writer.close()


JAXBoardLoggerActor = ray.remote(num_cpus=0)(JAXBoardLoggerV2)


class TerminalLogger(Logger):
    def __init__(self, name="logger", time_delta: float = 0.0):
        self._name = name
        self._time_delta = time_delta
        self._time = time.time()
        self._steps = 0

    def write(self, data: Union[List[LogDatum], LogDatum, dict]):
        data_ready = LogDatum.from_any(data)

        # TODO: refactor if this time delta is used at least three times
        now = time.time()
        if (now - self._time) > self._time_delta:
            for datum in data_ready:
                self._write_now_datum(datum)
            self._time = now

        self._steps += 1

    def _write_now_datum(self, datum: LogDatum):
        prefixed_key = self._name + ":" + datum.tag
        if isinstance(datum, LogText):
            print(f"{prefixed_key} = {datum.text}")
        elif isinstance(datum, LogScalar):
            print(f"{prefixed_key} = {datum.scalar}")


TerminalLoggerActor = ray.remote(num_cpus=0)(TerminalLogger)
