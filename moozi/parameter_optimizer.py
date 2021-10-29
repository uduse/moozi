from dataclasses import dataclass, field
from typing import Callable, List

import chex
import jax
import jax.numpy as jnp
import optax
import ray
import rlax
from absl import logging

import moozi as mz
from moozi.learner import TrainingState


@ray.remote
@dataclass
class LoggerActor:
    logger: mz.logging.Logger

    def log(self, data):
        self.logger.write(data)


def make_sgd_step_fn(
    network: mz.nn.NeuralNetwork,
    loss_fn: mz.loss.LossFn,
    optimizer,
    target_update_period: int = 1,
):
    # @partial(jax.jit, backend="cpu")
    @jax.jit
    @chex.assert_max_traces(n=1)
    def sgd_step_fn(state: TrainingState, batch: mz.replay.TrainTarget):
        # gradient descend
        _, new_key = jax.random.split(state.rng_key)
        grads, extra = jax.grad(loss_fn, has_aux=True, argnums=1)(
            network, state.params, batch
        )
        updates, new_opt_state = optimizer.update(grads, state.opt_state)
        new_params = optax.apply_updates(state.params, updates)
        new_steps = state.steps + 1

        target_params = rlax.periodic_update(
            new_params, state.target_params, new_steps, target_update_period
        )

        new_training_state = TrainingState(
            params=new_params,
            target_params=target_params,
            opt_state=new_opt_state,
            steps=new_steps,
            rng_key=new_key,
        )

        # KL calculation
        orig_logits = network.initial_inference(
            state.params, batch.stacked_frames
        ).policy_logits
        new_logits = network.initial_inference(
            new_params, batch.stacked_frames
        ).policy_logits
        prior_kl = jnp.mean(rlax.categorical_kl_divergence(orig_logits, new_logits))

        # store data to log
        step_data = mz.logging.JAXBoardStepData(scalars={}, histograms={})
        step_data.update(extra)
        step_data.scalars["prior_kl"] = prior_kl
        step_data.histograms["reward"] = batch.last_reward
        step_data.add_hk_params(new_params)

        return new_training_state, step_data

    return sgd_step_fn


@dataclass(repr=False)
class ParameterOptimizer:
    network: mz.nn.NeuralNetwork = field(init=False)
    state: TrainingState = field(init=False)
    sgd_step_fn: Callable = field(init=False)

    loggers: List[mz.logging.Logger] = field(default_factory=list)

    _num_updates: int = 0

    def build(self, factory):
        network, params, loss_fn, optimizer = factory()
        self.network = network
        self.state = TrainingState(
            params=params,
            opt_state=optimizer.init(params),
            steps=0,
            rng_key=jax.random.PRNGKey(0),
        )
        self.sgd_step_fn = make_sgd_step_fn(network, loss_fn, optimizer)

    def update(self, batch):
        if len(batch) == 0:
            raise ValueError("Batch is empty")
        self.state, step_data = self.sgd_step_fn(self.state, batch)
        self._num_updates += 1
        self._log_step_data(step_data)

    def get_params(self):
        return self.state.params

    def get_network(self):
        return self.network

    # def save(self, path):
    #     with open(path, "wb") as f:
    #         cloudpickle.dump(self, f)

    # @classmethod
    # def restore(path):
    #     with open(path, "rb") as f:
    #         return cloudpickle.load(f)

    def build_loggers(self, loggers_factory: Callable[[], List[mz.logging.Logger]]):
        self.loggers = loggers_factory()
        logging.info("setting loggers")

    def _log_step_data(self, step_data):
        for logger in self.loggers:
            if isinstance(logger, mz.logging.JAXBoardLogger):
                logger.write(step_data)
            else:
                raise NotImplementedError(f"Logger type {type(logger)} not supported")

    def get_stats(self):
        return dict(num_updates=self._num_updates)

    def log_stats(self):
        logging.info(self.get_stats())

    def close(self):
        for logger in self._loggers:
            if isinstance(logger, mz.logging.JAXBoardLogger):
                logging.info(logger._name, "closed")
                logger.close()
