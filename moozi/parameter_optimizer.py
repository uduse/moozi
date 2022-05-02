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
from moozi.core import Config, TrainingState, TrainTarget
from moozi.logging import LogDatum
from moozi.nn import RootFeatures
from moozi.nn.nn import NNModel, make_model


@ray.remote
@dataclass
class LoggerActor:
    logger: mz.logging.Logger

    def log(self, data):
        self.logger.write(data)


def _compute_prior_kl(
    model: NNModel, batch: TrainTarget, orig_params, new_params, state
):
    is_training = False
    orig_out, _ = model.root_inference(
        orig_params,
        state,
        RootFeatures(obs=batch.stacked_frames, player=jnp.array(0)),
        is_training,
    )
    orig_logits = orig_out.policy_logits
    new_out, _ = model.root_inference(
        new_params,
        state,
        RootFeatures(obs=batch.stacked_frames, player=jnp.array(0)),
        is_training,
    )
    new_logits = new_out.policy_logits
    prior_kl = jnp.mean(rlax.categorical_kl_divergence(orig_logits, new_logits))
    return prior_kl


def make_sgd_step_fn(
    model: mz.nn.NNModel,
    loss_fn: mz.loss.LossFn,
    optimizer,
    target_update_period: int = 1,
    include_prior_kl: bool = True,
):
    @jax.jit
    @chex.assert_max_traces(n=1)
    def sgd_step_fn(training_state: TrainingState, batch: mz.replay.TrainTarget):
        # gradient descend
        _, new_key = jax.random.split(training_state.rng_key)
        grads, extra = jax.grad(loss_fn, has_aux=True, argnums=1)(
            model, training_state.params, training_state.state, batch
        )
        updates, new_opt_state = optimizer.update(grads, training_state.opt_state)
        new_params = optax.apply_updates(training_state.params, updates)
        new_steps = training_state.steps + 1

        # TODO: put the target_update_period in the config and use it
        target_params = rlax.periodic_update(
            new_params, training_state.target_params, new_steps, target_update_period
        )

        new_training_state = TrainingState(
            params=new_params,
            target_params=target_params,
            state=extra["state"],
            opt_state=new_opt_state,
            steps=new_steps,
            rng_key=new_key,
        )

        step_data = extra["step_data"]

        for module, weight_name, weights in hk.data_structures.traverse(new_params):
            name = module + "/" + weight_name
            step_data[name] = weights

        if include_prior_kl:
            prior_kl = _compute_prior_kl(
                model, batch, training_state.params, new_params, training_state.state
            )
            step_data["prior_kl"] = prior_kl
        return new_training_state, dict(step_data=step_data)

    return sgd_step_fn


@dataclass(repr=False)
class ParameterOptimizer:
    model: mz.nn.NNModel = field(init=False)
    training_state: TrainingState = field(init=False)
    sgd_step_fn: Callable = field(init=False)
    is_remote: bool = False

    loggers: List[mz.logging.Logger] = field(default_factory=list)

    _last_step_data: List[LogDatum] = field(default_factory=list)
    _num_updates: int = 0

    def __post_init__(self):
        logger.remove()
        logger.add(sys.stderr, level="ERROR")
        logger.add("logs/param_opt.log", level="DEBUG")
        logger.info(f"Parameter optimizer created, {vars(self)}")

    @staticmethod
    def from_config(config: Config, remote: bool = False):
        if remote:
            param_opt = ray.remote(num_gpus=0.4)(ParameterOptimizer).remote(
                is_remote=True
            )
            param_opt.make_training_suite.remote(config)
            param_opt.make_loggers.remote(
                lambda: [
                    mz.logging.JAXBoardLoggerV2(name="param_opt", time_delta=15),
                ]
            )
            return param_opt
        else:
            param_opt = ParameterOptimizer()
            param_opt.make_training_suite(config)
            param_opt.make_loggers(
                lambda: [
                    mz.logging.JAXBoardLoggerV2(name="param_opt", time_delta=15),
                ]
            )
            return param_opt

    def make_training_suite(self, config: Config):
        self.model = make_model(config.nn_arch_cls, config.nn_spec)
        params, state = self.model.init_params_and_state(
            jax.random.PRNGKey(config.seed)
        )
        loss_fn = mz.loss.MuZeroLoss(
            num_unroll_steps=config.num_unroll_steps, weight_decay=config.weight_decay
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(1),
            optax.adam(config.lr, b1=0.9, b2=0.99),
        )
        self.training_state = TrainingState(
            params=params,
            target_params=params,
            state=state,
            opt_state=optimizer.init(params),
            steps=0,
            rng_key=jax.random.PRNGKey(config.seed),
        )
        self.sgd_step_fn = make_sgd_step_fn(self.model, loss_fn, optimizer)

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

        for i in range(0, len(train_targets), batch_size):
            batch_slice = train_targets[i : i + batch_size]
            if len(batch_slice) != batch_size:
                break
            batch = stack_sequence_fields(batch_slice)
            self.training_state, extra = self.sgd_step_fn(self.training_state, batch)
            self._num_updates += 1
            self._last_step_data = extra["step_data"]

        logger.debug(self.get_stats())

    def get_params_and_state(self) -> Union[ray.ObjectRef, Tuple[hk.Params, hk.State]]:
        logger.debug("getting params and state")
        ret = self.training_state.params, self.training_state.state
        if self.is_remote:
            return ray.put(ret)
        else:
            return ret

    def get_model(self):
        logger.debug("getting model")
        return self.model

    def save(self, path):
        logger.debug(f"saving model to {path}")
        with open(path, "wb") as f:
            cloudpickle.dump((self.training_state, self.model, self.n), f)

    def restore(self, path):
        logger.debug(f"restoring model from {path}")
        with open(path, "rb") as f:
            self.training_state, self.model, self.sgd_step_fn = cloudpickle.load(f)

    def make_loggers(self, loggers_factory: Callable[[], List[mz.logging.Logger]]):
        self.loggers = loggers_factory()
        logging.info("setting loggers" + str(self.loggers))

    def get_stats(self):
        return dict(num_updates=self._num_updates)

    def get_properties(self) -> dict:
        import os

        import jax

        ray_gpu_ids = ray.get_gpu_ids()
        cuda_visible_devices = os.environ["CUDA_VISIBLE_DEVICES"]
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

    def log(self):
        for logger in self.loggers:
            if isinstance(logger, mz.logging.JAXBoardLoggerV2):
                logger.write(
                    LogDatum.from_any(self.get_stats())
                    + LogDatum.from_any(self._last_step_data)
                )
            elif isinstance(logger, mz.logging.TerminalLogger):
                logger.write(self.get_stats())

    def close(self):
        for logger in self.loggers:
            if isinstance(logger, mz.logging.JAXBoardLoggerV2):
                logger.close()
