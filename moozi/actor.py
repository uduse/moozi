import typing
import random
from collections import defaultdict

import acme
import acme.jax.utils as acme_utils
import acme.jax.variable_utils
import acme.wrappers.open_spiel_wrapper
import chex
import dm_env
import jax
import jax.numpy as jnp
import numpy as np
import rlax

import moozi as mz


class RandomActor(acme.core.Actor):
    def __init__(self, adder):
        self._adder = adder

    def select_action(self, observation: acme.wrappers.open_spiel_wrapper.OLT) -> int:
        legals = np.array(np.nonzero(observation.legal_actions), dtype=np.int32)
        return np.random.choice(legals[0])

    def observe_first(self, timestep: dm_env.TimeStep):
        self._adder.add_first(timestep)

    def observe(self, action: chex.Array, next_timestep: dm_env.TimeStep):
        self._adder.add(action, next_timestep)

    def update(self, wait: bool = False):
        pass


class PriorPolicyActor(acme.core.Actor):
    r"""

    # NOTE: acme's actor's batching behavior is inconsistent
    # https://github.com/deepmind/acme/blob/aba3f195afd3e9774e2006ec9b32cb76048b7fe6/acme/agents/jax/actors.py#L82
    # TODO: replace vmap with manual batching?
    # https://github.com/deepmind/acme/blob/926b17ad116578801a0fbbe73c4ddc276a28e23e/acme/agents/jax/actors.py#L76
    # self._policy_fn = jax.jit(jax.vmap(_policy_fn, in_axes=[None, 0, 0, None]))

    """

    def __init__(
        self,
        environment_spec: acme.specs.EnvironmentSpec,
        network: mz.nn.NeuralNetwork,
        adder,
        variable_client: acme.jax.variable_utils.VariableClient,
        random_key,
        epsilon=0.1,
        temperature=1,
        loggers: typing.Optional[typing.List] = None,
        name: typing.Optional[str] = None,
    ):
        self._name = name or self.__class__.__name__
        self._env_spec = environment_spec
        self._random_key = random_key
        self._adder = adder
        self._client = variable_client
        self._policy_fn = self._make_policy_fn(network, epsilon, temperature)
        self._loggers = loggers or self._get_default_loggers()
        self._last_rewards = []
        self._last_actions = []

    def _get_default_loggers(self):
        return []

    def _make_policy_fn(self, network, epsilon, temperature):
        @jax.jit
        @chex.assert_max_traces(n=1)
        def _policy_fn(
            params, image: chex.Array, legal_actions_mask: chex.Array, random_key
        ) -> typing.Tuple[chex.Array, mz.logging.JAXBoardStepData]:
            chex.assert_rank(image, 1)
            chex.assert_shape(legal_actions_mask, [self._env_spec.actions.num_values])

            network_output = network.initial_inference(
                params, acme_utils.add_batch_dim(image)
            )
            action_logits = acme_utils.squeeze_batch_dim(network_output.policy_logits)
            chex.assert_rank(action_logits, 1)
            action_entropy = rlax.softmax().entropy(action_logits)
            chex.assert_rank(action_entropy, 0)
            _sampler = rlax.epsilon_softmax(epsilon, temperature).sample
            action = _sampler(random_key, action_logits)

            step_data = mz.logging.JAXBoardStepData({}, {})
            step_data.add_hk_params(params)
            step_data.scalars["action_entropy"] = action_entropy
            step_data.histograms["action_logits"] = action_logits
            return action, step_data

        return _policy_fn

    def select_action(self, observation: acme.wrappers.open_spiel_wrapper.OLT) -> int:
        self._random_key, new_key = jax.random.split(self._random_key)
        action, step_data = self._policy_fn(
            self._client.params,
            image=observation.observation,
            legal_actions_mask=observation.legal_actions,
            random_key=new_key,
        )
        self._log(step_data)
        self._last_actions.append(action)
        self._last_actions = self._last_actions[-1000:]
        return action

    def _log(self, data: mz.logging.JAXBoardStepData):
        for logger in self._loggers:
            if isinstance(logger, mz.logging.JAXBoardLogger):
                if len(self._last_rewards) >= 1000:
                    data.scalars["rolling_reward"] = np.mean(self._last_rewards)
                    data.histograms["last_rewards"] = self._last_rewards
                    data.histograms["last_actions"] = self._last_actions
                logger.write(data)

    def observe_first(self, timestep: dm_env.TimeStep):
        # timestep.observation = {"env": timestep.observation, "search": None}
        self._adder.add_first(timestep)

    def observe(self, action: chex.Array, next_timestep: dm_env.TimeStep):
        if next_timestep.last():
            self._last_rewards.append(next_timestep.reward)
            self._last_rewards = self._last_rewards[-1000:]
        # next_timestep.observation = {"env": next_timestep.observation, "search": None}
        self._adder.add(action, next_timestep)

    def update(self, wait: bool = False):
        self._client.update(wait)

    def close(self):
        for logger in self._loggers:
            if isinstance(logger, mz.logging.JAXBoardLogger):
                print(logger._name, "closed")
                logger.close()

    def __del__(self):
        self.close()


class ActorCache(object):
    r"""A collection of queues that keep the most recent values."""

    def __init__(self, size: int = 1000) -> None:
        assert size >= 1
        self._size = size
        self._data = defaultdict(list)

    def put(self, key, value):
        self._data[key].append(value)
        self._data[key] = self._data[key][-self._size :]

    def get_all(self, key):
        assert key in self._data
        return self._data[key]

    def get_last(self, key):
        assert key in self._data
        return self._data[key][-1]


class RandomActorWithAdder(acme.core.Actor):
    def __init__(
        self,
        environment_spec: acme.specs.EnvironmentSpec,
        adder: mz.replay.MooZiAdder,
        name: typing.Optional[str] = None,
    ):
        self._env_spec = environment_spec
        self._name = name or self.__class__.__name__
        self._adder = adder
        self._cache = ActorCache()

    def select_action(self, observation: acme.wrappers.open_spiel_wrapper.OLT) -> int:
        # generate search stats
        root_value = np.random.normal()
        num_actions = self._env_spec.actions.num_values
        action_probs = np.random.normal(size=num_actions)
        action_probs = (action_probs - action_probs.min()) / action_probs.sum()
        child_visits = action_probs

        self._cache.put("root_value", root_value)
        self._cache.put("child_vists", child_visits)

        action = random.choice(list(range(num_actions)))
        return action

    def observe_first(self, timestep: dm_env.TimeStep):
        observation = mz.replay.Observation.from_env_timestep(timestep)
        self._adder.add_first(observation)

    def observe(self, action: chex.Array, next_timestep: dm_env.TimeStep):
        assert self._cache.get_last("action") == action
        root_value = self._cache.get_last("root_value")
        child_visits = self._cache.get_last("child_visits")
        last_timestep_info = mz.replay.Work(action, root_value, child_visits)
        self._adder.add(last_timestep_info, next_timestep)

    def update(self, wait: bool = False):
        self._client.update(wait)
