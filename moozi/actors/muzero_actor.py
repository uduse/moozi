from collections import defaultdict
import functools
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
from moozi.policies import PolicyFeed, PolicyFn


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

    def __len__(self):
        return len(self._list)

    @property
    def size(self):
        return self._size


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
        variable_client: VariableClient,
        policy_fn: PolicyFn,
        adder: mz.replay.Adder,
        random_key,
        num_stacked_frames: int = 8,
        loggers: Optional[List] = None,
        name: Optional[str] = None,
    ):
        self._env_spec = env_spec
        self._variable_client = variable_client
        self._policy = policy_fn
        self._adder = adder
        self._loggers = loggers or []
        self._name = name or self.__class__.__name__
        self._num_stacked_frames = num_stacked_frames

        def _init_memory():
            return {
                "random_key": random_key,
                "last_frames": SimpleQueue(5000),
                "rolling_rewards": SimpleQueue(5000),
                "policy_results": SimpleQueue(5000),
                "action_probs": SimpleQueue(5000),
            }

        self._init_memory_fn = _init_memory
        self._memory = self._init_memory_fn()

    def reset_memory(self):
        self._memory = self._init_memory_fn()

    def select_action(self, observation: OLT) -> int:
        last_frames = self.m["last_frames"].get()[-self._num_stacked_frames :]
        while len(last_frames) < self._num_stacked_frames:
            padding = np.zeros_like(observation.observation)
            last_frames.append(padding)
        obs_stacked_frames = jnp.array(last_frames)

        key, new_key = jax.random.split(self.m["random_key"])
        self.m["random_key"] = key
        policy_feed = PolicyFeed(
            stacked_frames=obs_stacked_frames,
            legal_actions_mask=jnp.array(observation.legal_actions),
            random_key=new_key,
        )
        policy_result = self._policy(self._variable_client.params, policy_feed)
        self.m["policy_results"].put(policy_result)
        action_space_size = self._env_spec.actions.num_values
        dummy_action_probs = np.zeros(action_space_size, dtype=np.float32)
        self.m["action_probs"].put(
            policy_result.extras.get("action_probs", dummy_action_probs)
        )
        return policy_result.action

    def observe_first(self, timestep: dm_env.TimeStep):
        if isinstance(timestep.observation, OLT):
            self.m["last_frames"].put(timestep.observation.observation)
        else:
            raise NotImplementedError
        observation = mz.replay.Observation.from_env_timestep(timestep)
        self._adder.add_first(observation)

    def observe(self, action: chex.Array, next_timestep: dm_env.TimeStep):
        if isinstance(next_timestep.observation, OLT):
            self.m["last_frames"].put(next_timestep.observation.observation)
        else:
            raise NotImplementedError
        root_value, action_probs = self._get_last_search_stats()
        last_reflection = mz.replay.Reflection(action, root_value, action_probs)
        next_observation = mz.replay.Observation.from_env_timestep(next_timestep)
        rolling_reward = self._update_rolling_reward(next_observation)
        data = mz.logging.JAXBoardStepData(
            scalars={"rolling_rewards": rolling_reward}, histograms={}
        )
        self._log(data)
        self._adder.add(last_reflection, next_observation)

    def _update_rolling_reward(self, next_observation):
        if next_observation.is_last:
            self.m["rolling_rewards"].put(next_observation.reward)
        rolling_rewards = self.m["rolling_rewards"].get()
        if rolling_rewards:
            rolling_rewards = np.mean(rolling_rewards)
        else:
            rolling_rewards = 0.0
        return rolling_rewards

    def _get_last_search_stats(self):
        latest_policy_extras = self.m["policy_results"].get()[-1].extras
        root_value = latest_policy_extras.get("root_value", np.float32(0))
        action_space_size = self._env_spec.actions.num_values
        dummy_action_probs = np.zeros(action_space_size, dtype=np.float32)
        action_probs = latest_policy_extras.get("action_probs", dummy_action_probs)
        return root_value, action_probs

    def _log(self, data: mz.logging.JAXBoardStepData):
        for logger in self._loggers:
            if isinstance(logger, mz.logging.JAXBoardLogger):
                logger.write(data)

    def update(self, wait: bool = False):
        self._variable_client.update(wait)

    def close(self):
        for logger in self._loggers:
            if isinstance(logger, mz.logging.JAXBoardLogger):
                print(logger._name, "closed")
                logger.close()

    @property
    def m(self):
        return self._memory
