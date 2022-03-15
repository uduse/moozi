from dataclasses import dataclass, field
import haiku as hk
from re import I
import cloudpickle
from pprint import pprint
from typing import Callable, List

import chex
import jax
import jax.numpy as jnp
import optax
import ray
import rlax
from absl import logging

import moozi as mz
from moozi.core import Config, make_env_spec, make_env
from moozi.learner import TrainingState
from moozi.logging import LogHistogram, LogDatum, LogScalar
from moozi.nn import RootFeatures
from moozi.nn.nn import NNModel, make_model
from moozi.replay import TrainTarget


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
        RootFeatures(stacked_frames=batch.stacked_frames, player=jnp.array(0)),
        is_training,
    )
    orig_logits = orig_out.policy_logits
    new_out, _ = model.root_inference(
        new_params,
        state,
        RootFeatures(stacked_frames=batch.stacked_frames, player=jnp.array(0)),
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

        return new_training_state, dict(state=extra["state"], step_data=step_data)

    return sgd_step_fn


@dataclass(repr=False)
class ParameterOptimizer:
    model: mz.nn.NNModel = field(init=False)
    training_state: TrainingState = field(init=False)
    sgd_step_fn: Callable = field(init=False)

    loggers: List[mz.logging.Logger] = field(default_factory=list)
    _last_step_data: List[LogDatum] = field(default_factory=list)

    _num_updates: int = 0

    def make_training_suite(self, config: Config):
        self.model = make_model(config.nn_arch_cls, config.nn_spec)
        params, state = self.model.init_params_and_state(jax.random.PRNGKey(0))
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
            rng_key=jax.random.PRNGKey(0),
        )
        self.sgd_step_fn = make_sgd_step_fn(self.model, loss_fn, optimizer)

    def update(self, batch: mz.replay.TrainTarget):
        if len(batch) == 0:
            raise ValueError("Batch is empty")
        self.training_state, extra = self.sgd_step_fn(self.training_state, batch)
        self._num_updates += 1
        self._last_step_data = extra["step_data"]

    def get_params_and_state(self):
        # TODO: use ray.put instead
        return (self.training_state.params, self.training_state.state)

    def get_model(self):
        return self.model

    def save(self, path):
        with open(path, "wb") as f:
            cloudpickle.dump(self.training_state, f)

    def restore(self, path):
        # NOTE: not tested
        with open(path, "rb") as f:
            self.training_state = cloudpickle.load(f)

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
        model_size_str = f"{model_size_in_bytes / 1e6:.2f} MB"

        return dict(
            ray_gpu_ids=ray_gpu_ids,
            cuda_visible_devices=cuda_visible_devices,
            jax_devices=str(jax_devices),
            model_size=model_size_str,
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
        for logger in self._loggers:
            if isinstance(logger, mz.logging.JAXBoardLoggerV2):
                logging.info(logger._name, "closed")
                logger.close()
