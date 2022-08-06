import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Union

import haiku as hk
import jax.numpy as jnp
import numpy as np
import ray
from loguru import logger

import moozi as mz


@dataclass
class LogDatum:
    tag: str

    @staticmethod
    def from_dict(d: dict) -> List["LogDatum"]:
        # TODO: this is a little bit stupid, we should register dataclasses as pytree
        assert isinstance(d, dict)
        mappers = (
            (float, LogScalar),
            (int, LogScalar),
            (str, LogText),
            (np.array, LogHistogram),
        )

        def process(key, val):
            if isinstance(val, (int, float)):
                return LogScalar(key, val)

            try:
                return LogScalar(key, val.item())
            except:
                pass

            if isinstance(val, (jnp.ndarray, np.ndarray)):
                if val.size == 1:
                    return LogScalar(key, float(val))
                else:
                    return LogHistogram(key, np.array(val))

            for cast_fn, target_cls in mappers:
                try:
                    val = cast_fn(val)
                except (ValueError, TypeError, OverflowError):
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
        elif isinstance(data, (list, tuple)):
            return sum([LogDatum.from_any(d) for d in data], [])
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
        self._log_dir = log_dir or "./tb/"
        self._log_dir = str(Path(self._log_dir).resolve())
        self._time_delta = time_delta
        self._time = time.time()
        self._steps = 0
        self._writer = mz.jaxboard.SummaryWriter(name, log_dir=self._log_dir)
        logger.info(f"{self._name} is logging to {(self._log_dir)}")

    def write(
        self, data: Union[List[LogDatum], LogDatum, dict], step: Optional[int] = None
    ):
        data_ready = LogDatum.from_any(data)

        now = time.time()
        if (now - self._time) > self._time_delta:
            for datum in data_ready:
                self._write_now_datum(datum, step)
            self._time = now
            self._writer.flush()

        self._steps += 1

    def set_steps(self, steps):
        self._steps = steps

    def _write_now_datum(self, datum: LogDatum, step: Optional[int] = None):
        if step is None:
            step = self._steps

        if isinstance(datum, LogText):
            self._writer.text(tag=datum.tag, textdata=datum.text, step=step)
        elif isinstance(datum, LogImage):
            self._writer.image(tag=datum.tag, image=datum.image, step=step)
        elif isinstance(datum, LogScalar):
            self._writer.scalar(tag=datum.tag, value=datum.scalar, step=step)
        elif isinstance(datum, LogHistogram):
            num_bins = 50
            self._writer.histogram(
                tag=datum.tag,
                values=datum.values,
                bins=num_bins,
                step=step,
            )
        else:
            raise ValueError(f"Unsupported datum type: {type(datum)}")

    def close(self):
        r"""
        Always call this method the logging is finished.
        Otherwise unexpected hangings may occur.
        """
        return self._writer.close()


# TODO: deprecated the use of terminal logger, just log to files
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
        if isinstance(datum, LogText):
            print(f"{datum.tag} = {datum.text}")
        elif isinstance(datum, LogScalar):
            print(f"{datum.tag} = {datum.scalar}")


JAXBoardLoggerRemote = ray.remote(num_cpus=0)(JAXBoardLoggerV2)
TerminalLoggerRemote = ray.remote(num_cpus=0)(TerminalLogger)


def describe_np_array(arr, name):
    return {
        name + "/size": np.size(arr),
        name + "/min": np.min(arr),
        name + "/mean": np.mean(arr),
        name + "/max": np.max(arr),
        name + "/std": np.std(arr),
    }
