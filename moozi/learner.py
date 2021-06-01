import typing

import acme
import acme.jax.utils
import chex
import jax
import jax.numpy as jnp
import optax

import moozi as mz
import chex


class TrainingState(typing.NamedTuple):
    params: chex.ArrayTree
    opt_state: optax.OptState
    steps: int
    rng_key: jax.random.PRNGKey


class LearnsNothingLearner(acme.Learner):
    def __init__(self, params, *args, **kwargs):
        self.params = params

    def step(self):
        pass

    def get_variables(self, names):
        return [self.params]

    def save(self):
        return self.params

    def restore(self, state):
        self.params = state


class RandomNoiseLearner(acme.Learner):
    def __init__(self, params, *args, **kwargs):
        self.params = params
        self.key = jax.random.PRNGKey(0)

    def step(self):
        self.key, new_key = jax.random.split(self.key)
        self.params = jax.tree_map(
            lambda x: x + jax.random.normal(new_key), self.params
        )

    def get_variables(self, names):
        return [self.params]

    def save(self):
        return self.params

    def restore(self, state):
        self.params = state


class SGDLearner(acme.Learner):
    def __init__(
        self,
        network: mz.nn.NeuralNetwork,
        loss_fn: mz.loss.LossFn,
        optimizer: optax.GradientTransformation,
        data_iterator,
        random_key,
        loggers: typing.Optional[typing.List] = None,
        name: typing.Optional[str] = None,
    ):
        self._name = name or self.__class__.__name__
        self._counter = acme.utils.counting.Counter()
        self._loggers = loggers or self._get_default_loggers()

        self._data_iterator = acme.jax.utils.prefetch(data_iterator)

        self._network = network
        self._sgd_step_fn = self._make_sgd_step_fn(network, loss_fn, optimizer)
        key_params, key_state = jax.random.split(random_key, 2)
        params = self._network.init(key_params)
        self._state = TrainingState(
            params=params, opt_state=optimizer.init(params), steps=0, rng_key=key_state
        )

    def _get_default_loggers(self):
        return [
            acme.utils.loggers.TerminalLogger(time_delta=5.0, print_fn=print),
            # mz.logging.JAXBoardLogger(self._name, time_delta=5.0),
        ]

    def _make_sgd_step_fn(self, network, loss_fn, optimizer):
        @jax.jit
        @chex.assert_max_traces(n=1)
        def _sgd_step(training_state: TrainingState, batch):
            # key, new_key = jax.random.split(training_state.rng_key)  # curently not using the key
            new_key = training_state.rng_key  # curently not using the key
            step_data = mz.logging.JAXBoardStepData(scalars={}, histograms={})
            grads, extra = jax.grad(loss_fn, has_aux=True)(
                network, training_state.params, batch
            )
            step_data.update(extra)
            updates, new_opt_state = optimizer.update(grads, training_state.opt_state)
            new_params = optax.apply_updates(training_state.params, updates)
            steps = training_state.steps + 1
            new_training_state = TrainingState(
                new_params, new_opt_state, steps, new_key
            )
            step_data.add_hk_params(new_params)
            step_data.histograms.update({"reward": batch.data.reward})
            return new_training_state, step_data

        # TODO: multiple SGDs per training step, not tested for now
        # def _postprocess_aux(extra: mz.logging.JAXBoardStepData):
        #     return extra._replace(scalars=jax.tree_map(jnp.mean, extra.scalars))

        # _sgd_step_one_batch = acme.jax.utils.process_multiple_batches(
        #     _sgd_step_one_batch, num_batches=1, postprocess_aux=_postprocess_aux
        # )
        return _sgd_step

    def step(self):
        batch = next(self._data_iterator)
        self._state, extra = self._sgd_step_fn(self._state, batch)
        result = self._counter.increment(steps=1)
        result.update(extra.scalars)

        for logger in self._loggers:
            if isinstance(logger, acme.utils.loggers.TerminalLogger):
                logger.write(result)
            elif isinstance(logger, mz.logging.JAXBoardLogger):
                logger.write(extra)

    def get_variables(self, names):
        return [self._state.params]

    def save(self):
        return self._state

    def restore(self, state):
        self._state = state
