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
from acme.core import Actor
from acme.jax.utils import add_batch_dim, squeeze_batch_dim
from acme.jax.variable_utils import VariableClient
from acme.utils import tree_utils
from acme.wrappers.open_spiel_wrapper import OLT
from moozi.policies import PolicyFeed, PolicyFn
from nptyping import NDArray


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


class MooZiActor(Actor):
    r"""

    # NOTE: acme's actor's batching behavior is inconsistent
    # https://github.com/deepmind/acme/blob/aba3f195afd3e9774e2006ec9b32cb76048b7fe6/acme/agents/jax/actors.py#L82
    # TODO: replace vmap with manual batching?
    # https://github.com/deepmind/acme/blob/926b17ad116578801a0fbbe73c4ddc276a28e23e/acme/agents/jax/actors.py#L76
    # self._policy_fn = jax.jit(jax.vmap(_policy_fn, in_axes=[None, 0, 0, None]))

    """

    def __init__(
        self,
        environment_spec: specs.EnvironmentSpec,
        network: mz.nn.NeuralNetwork,
        adder,
        variable_client: VariableClient,
        random_key,
        policy_fn: PolicyFn,
        loggers: Optional[List] = None,
        name: Optional[str] = None,
    ):
        self._name = name or self.__class__.__name__
        self._env_spec = environment_spec
        self._random_key = random_key
        self._adder = adder
        self._client = variable_client
        self._loggers = loggers
        self._policy_fn = policy_fn

        self._memory = {"last_frames": SimpleQueue(5), "random_key": random_key}

    def select_action(self, observation: OLT) -> int:
        if isinstance(observation, OLT):
            self._memory["last_frames"].put(observation.observation)

        stacked_frames = jnp.array(self._memory["last_frames"].get())

        policy_feed = PolicyFeed(
            params=self._client.params,
            stacked_frames=stacked_frames,
            legal_actions_mask=jnp.array(observation.legal_actions),
            random_key=self._memory["random_key"],
        )
        result = self._policy_fn.run(policy_feed)
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

    # def _log(self, data: mz.logging.JAXBoardStepData):
    #     for logger in self._loggers:
    #         if isinstance(logger, mz.logging.JAXBoardLogger):
    #             if len(self._last_rewards) >= 1000:
    #                 data.scalars["rolling_reward"] = np.mean(self._last_rewards)
    #                 data.histograms["last_rewards"] = self._last_rewards
    #                 data.histograms["last_actions"] = self._last_actions
    #             logger.write(data)

    def observe_first(self, timestep: dm_env.TimeStep):
        observation = mz.replay.Observation.from_env_timestep(timestep)
        self._adder.add_first(observation)

    def observe(self, action: chex.Array, next_timestep: dm_env.TimeStep):
        assert self._cache.get_last("action") == action
        root_value = self._cache.get_last("root_value")
        child_visits = self._cache.get_last("child_visits")
        last_timestep_info = mz.replay.Reflection(action, root_value, child_visits)
        self._adder.add(last_timestep_info, next_timestep)

    def update(self, wait: bool = False):
        self._client.update(wait)

    def close(self):
        for logger in self._loggers:
            if isinstance(logger, mz.logging.JAXBoardLogger):
                print(logger._name, "closed")
                logger.close()

    def __del__(self):
        self.close()
