from collections import defaultdict
from typing import Dict, List, NamedTuple, Optional, Tuple, overload

import chex
import dm_env
import jax
import jax.numpy as jnp
import moozi as mz
import numpy as np
import rlax
import tree
from acme import specs, types
from acme.core import Actor as BaseActor
from acme.jax.utils import add_batch_dim, squeeze_batch_dim
from acme.jax.variable_utils import VariableClient
from acme.utils import tree_utils
from acme.wrappers.open_spiel_wrapper import OLT
from moozi.policies import PolicyFeed, Policy


class SimpleQueue(object):
    r"""A simple FIFO queue."""

    def __init__(self, size: int = 1000) -> None:
        self._list: list = []
        self._size = size

    def put(self, value):
        self._list.append(value)
        if len(self._list) > self._size:
            self._list = self._list[-self._size :]

    def get(self):
        return self._list

    def is_full(self) -> bool:
        return len(self._list) == self._size


class MuZeroActor(BaseActor):
    r"""

    # NOTE: acme's actor's batching behavior is inconsistent
    # https://github.com/deepmind/acme/blob/aba3f195afd3e9774e2006ec9b32cb76048b7fe6/acme/agents/jax/actors.py#L82
    # TODO: replace vmap with manual batching?
    # https://github.com/deepmind/acme/blob/926b17ad116578801a0fbbe73c4ddc276a28e23e/acme/agents/jax/actors.py#L76
    # self._policy_fn = jax.jit(jax.vmap(_policy_fn, in_axes=[None, 0, 0, None]))

    """

    def __init__(
        self,
        env_spec: specs.EnvironmentSpec,
        policy: Policy,
        adder: mz.replay.Adder,
        random_key,
        num_stacked_frames: int = 8,
        loggers: Optional[List] = None,
        name: Optional[str] = None,
    ):
        self._env_spec = env_spec
        self._policy = policy
        self._adder = adder
        self._loggers = loggers or []
        self._name = name or self.__class__.__name__
        self._policy = policy

        self._memory = {
            "last_frames": SimpleQueue(num_stacked_frames),
            "rolling_average_reward": SimpleQueue(5000),
            "random_key": random_key,
            "policy_extras": SimpleQueue(5),
        }

    def select_action(self, observation: OLT) -> int:
        if isinstance(observation, OLT):
            while not self._memory["last_frames"].is_full():
                padding = np.zeros_like(observation.observation)
                self._memory["last_frames"].put(padding)
            self._memory["last_frames"].put(observation.observation)
        else:
            raise NotImplementedError

        stacked_frames = jnp.array(self._memory["last_frames"].get())

        key, new_key = jax.random.split(self._memory["random_key"])
        self._memory["random_key"] = key
        policy_feed = PolicyFeed(
            stacked_frames=stacked_frames,
            legal_actions_mask=jnp.array(observation.legal_actions),
            random_key=new_key,
        )
        result = self._policy.run(policy_feed)
        self._memory["policy_extras"].put(result.extras)
        return result.action

        # self._random_key, new_key = jax.random.split(self._random_key)
        # action, step_data = self._policy_fn(
        #     self._client.params,
        #     image=observation.observation,
        #     legal_actions_mask=observation.legal_actions,
        #     random_key=new_key,
        # )
        # self._log(step_data)
        # self._last_actions.append(action)
        # self._last_actions = self._last_actions[-1000:]
        # return action

    def _log(self, data: mz.logging.JAXBoardStepData):
        for logger in self._loggers:
            if isinstance(logger, mz.logging.JAXBoardLogger):
                logger.write(data)

    def observe_first(self, timestep: dm_env.TimeStep):
        observation = mz.replay.Observation.from_env_timestep(timestep)
        self._adder.add_first(observation)

    def observe(self, action: chex.Array, next_timestep: dm_env.TimeStep):
        root_value, child_visits = self._get_last_search_stats()
        last_reflection = mz.replay.Reflection(action, root_value, child_visits)
        next_observation = mz.replay.Observation.from_env_timestep(next_timestep)
        self._memory["rolling_average_reward"].put(next_observation.reward)
        rolling_reward = np.mean(self._memory["rolling_average_reward"].get())
        data = mz.logging.JAXBoardStepData(
            scalars={"rolling_reward": rolling_reward}, histograms={}
        )
        self._log(data)
        self._adder.add(last_reflection, next_observation)

    def _get_last_search_stats(self):
        action_space_size = self._env_spec.actions.num_values
        latest_policy_extras = self._memory["policy_extras"].get()[-1]
        root_value = latest_policy_extras.get("root_value", np.float32(0))
        dummy_child_visits = np.zeros(action_space_size, dtype=np.float32)
        child_visits = latest_policy_extras.get("child_visits", dummy_child_visits)
        return root_value, child_visits

    def update(self, wait: bool = False):
        self._policy.update(wait)

    def close(self):
        for logger in self._loggers:
            if isinstance(logger, mz.logging.JAXBoardLogger):
                print(logger._name, "closed")
                logger.close()

    def __del__(self):
        self.close()
