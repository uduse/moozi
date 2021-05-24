import functools
import typing

import acme
import acme.jax.utils
import chex
import jax
import jax.numpy as jnp
import optax

import moozi as mz


class TrainingState(typing.NamedTuple):
    params: typing.Any
    opt_state: optax.OptState
    steps: int
    rng_key: jax.random.PRNGKey


class LearnsNothingLearner(acme.Learner):
    def __init__(self, *args, **kwargs):
        pass

    def step(self):
        pass

    def get_variables(self, names):
        pass

    def save(self):
        pass

    def restore(self, state):
        pass


class MooZiLearner(acme.Learner):
    def __init__(self, network, loss_fn, optimizer, data_iterator, random_key):
        self.network = network
        self._loss = jax.jit(functools.partial(loss_fn, self.network))

        @jax.jit
        @chex.assert_max_traces(n=1)
        def _sgd_step_one_batch(training_state: TrainingState, batch):
            # key, new_key = jax.random.split(training_state.rng_key)  # curently not using the key
            new_key = training_state.rng_key  # curently not using the key

            (loss, extra), grads = jax.value_and_grad(self._loss, has_aux=True)(
                training_state.params, batch
            )
            extra.metrics.update({"loss": loss})
            updates, new_opt_state = optimizer.update(grads, training_state.opt_state)
            new_params = optax.apply_updates(training_state.params, updates)
            steps = training_state.steps + 1
            new_training_state = TrainingState(
                new_params, new_opt_state, steps, new_key
            )
            return new_training_state, extra

        def _postprocess_aux(extra: mz.loss.LossExtra):
            return extra._replace(metrics=jax.tree_map(jnp.mean, extra.metrics))

        self._sgd_step = acme.jax.utils.process_multiple_batches(
            _sgd_step_one_batch, num_batches=1, postprocess_aux=_postprocess_aux
        )
        self._data_iterator = acme.jax.utils.prefetch(data_iterator)

        key_params, key_state = jax.random.split(random_key, 2)
        params = self.network.init(key_params)
        self._state = TrainingState(
            params=params, opt_state=optimizer.init(params), steps=0, rng_key=key_state
        )
        self._counter = acme.utils.counting.Counter()
        self._logger = acme.utils.loggers.TerminalLogger(time_delta=1.0, print_fn=print)

    def step(self):
        batch = next(self._data_iterator)
        self._state, extra = self._sgd_step(self._state, batch)
        result = self._counter.increment(steps=1)
        result.update(extra.metrics)
        self._logger.write(result)

    def get_variables(self, names):
        return [self._state.params]

    def save(self):
        return self._state

    def restore(self, state):
        self._state = state
