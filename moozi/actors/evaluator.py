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

class Evaluator(BaseActor):
    def __init__(
        self,
        variable_client: VariableClient,
        policy_fn: PolicyFn,
        random_key,
        dim_action: int,
        num_stacked_frames: int = 8,
    ):
        self._variable_client = variable_client
        self._policy = policy_fn
        self._num_stacked_frames = num_stacked_frames
        self._dim_action = dim_action
        self._seed_random_key = random_key

        self._memory = {"random_key": None, "last_frames": None}
        self.reset_memory()

    def reset_memory(self):
        self._memory = {
            "random_key": self._seed_random_key,
            "last_frames": mz.utils.SimpleBuffer(5000),
            "policy_results": mz.utils.SimpleBuffer(5000),
        }

    def select_action(self, observation: OLT) -> int:
        last_frames = self.m["last_frames"].get()[-self._num_stacked_frames :]
        while len(last_frames) < self._num_stacked_frames:
            padding = np.zeros_like(observation.observation)
            last_frames.append(padding)
        obs_stacked_frames = np.array(last_frames)

        key, new_key = jax.random.split(self.m["random_key"])
        policy_feed = PolicyFeed(
            stacked_frames=obs_stacked_frames,
            legal_actions_mask=np.array(observation.legal_actions),
            random_key=new_key,
        )

        policy_result = self._policy(self._variable_client.params, policy_feed)

        self.m["policy_results"].put(policy_result)
        self.m["random_key"] = key

        return policy_result.action

    def observe_first(self, timestep: dm_env.TimeStep):
        if isinstance(timestep.observation, OLT):
            self.m["last_frames"].put(timestep.observation.observation)
        else:
            raise NotImplementedError

    def observe(self, action: np.ndarray, next_timestep: dm_env.TimeStep):
        if isinstance(next_timestep.observation, OLT):
            self.m["last_frames"].put(next_timestep.observation.observation)
        else:
            raise NotImplementedError

    def update(self, wait: bool = False):
        self._variable_client.update(wait)

    @property
    def m(self):
        return self._memory
