from pathlib import Path
import random
from dataclasses import dataclass, field
import sys
from typing import Callable, List, Optional, Tuple, Union

import chex
import cloudpickle
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import ray
import rlax
from absl import logging
from acme.utils.tree_utils import stack_sequence_fields, unstack_sequence_fields
from loguru import logger

import moozi as mz
from moozi.core import TrainingState, TrainTarget
from moozi.logging import LogDatum
from moozi.nn.nn import NNArchitecture, NNModel, NNSpec
from moozi.nn.training import make_training_suite


# @ray.remote
# @dataclass
# class LoggerActor:
#     logger: mz.logging.Logger

#     def log(self, data):
#         self.logger.write(data)


class ParameterServer:
    def __init__(
        self,
        training_suite_factory,
        use_remote=False,
        save_dir: str = "checkpoints/",
    ):
        self.model: NNModel
        self.training_state: TrainingState
        self.sgd_step_fn: Callable
        self.model, self.training_state, self.sgd_step_fn = training_suite_factory()
        self.use_remote: bool = use_remote
        self.save_dir: Path = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._tb_logger = mz.logging.JAXBoardLoggerV2(
            name="param_server", time_delta=30
        )
        self._last_step_data: dict = {}

        jax.config.update("jax_debug_nans", True)
        logger.remove()
        logger.add(sys.stderr, level="SUCCESS")
        logger.add(f"logs/ps.debug.log", level="DEBUG")
        logger.add(f"logs/ps.info.log", level="INFO")
        self._flog = logger

    def update(self, big_batch: TrainTarget, batch_size: int):
        if len(big_batch) == 0:
            logger.error("Batch is empty, update() skipped.")
            return

        train_targets: List[TrainTarget] = unstack_sequence_fields(
            big_batch, big_batch[0].shape[0]
        )
        if len(train_targets) % batch_size != 0:
            logger.warning(
                f"Batch size {batch_size} is not a divisor of the batch size {len(train_targets)}"
            )

        logger.debug(
            f"updating with {len(train_targets)} samples, batch size {batch_size}"
        )

        ret = {}
        for i in range(0, len(train_targets), batch_size):
            batch_slice = train_targets[i : i + batch_size]
            if len(batch_slice) != batch_size:
                break
            batch = stack_sequence_fields(batch_slice)
            self.training_state, extra = self.sgd_step_fn(self.training_state, batch)
            self._last_step_data = extra
            ret = {"loss": extra["loss"]}
        return ret

    def log_tensorboard(self, step: Optional[int] = None):
        log_datum = LogDatum.from_any(self._last_step_data)
        if step is None:
            step = self.training_state.steps
        self._tb_logger.write(log_datum, step=step)
        if "loss" in self._last_step_data:
            self._flog.info(f"loss: {self._last_step_data['loss']}")
        return log_datum

    def get_training_steps(self):
        return self.training_state.steps

    def get_params(self):
        logger.debug("getting params")
        ret = self.training_state.target_params
        if self.use_remote:
            return ray.put(ret)
        else:
            return ret

    def get_state(self):
        logger.debug("getting state")
        ret = self.training_state.target_state
        if self.use_remote:
            return ray.put(ret)
        else:
            return ret

    def set_state(self, state):
        logger.debug("setting state")
        self.training_state.state = state

    def get_model(self):
        logger.debug("getting model")
        return self.model

    def get_device_properties(self) -> dict:
        import os
        import jax

        ray_gpu_ids = ray.get_gpu_ids()
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        jax_devices = jax.devices()

        model_size_in_bytes = hk.data_structures.tree_size(
            (self.training_state.params, self.training_state.state)
        )
        model_size_human = f"{model_size_in_bytes / 1e6:.2f} MB"

        return dict(
            ray_gpu_ids=ray_gpu_ids,
            cuda_visible_devices=cuda_visible_devices,
            jax_devices=str(jax_devices),
            model_size=model_size_human,
        )

    def save(self):
        path = self.save_dir / f"{self.training_state.steps}.pkl"
        logger.info(f"saving model to {path}")
        with open(path, "wb") as f:
            cloudpickle.dump((self.model, self.training_state, self.sgd_step_fn), f)

    def restore(self, path):
        logger.info(f"restoring model from {path}")
        with open(path, "rb") as f:
            self.model, self.training_state, self.sgd_step_fn = cloudpickle.load(f)
