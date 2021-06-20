from collections import defaultdict
from functools import partial
from typing import Dict, List, NamedTuple, Optional, Tuple

import chex
import dm_env
import jax
import jax.numpy as jnp
import moozi as mz
import numpy as np
import rlax
from acme import specs, types
from acme.core import Actor
from acme.jax.utils import add_batch_dim, squeeze_batch_dim
from acme.jax.variable_utils import VariableClient
from acme.wrappers.open_spiel_wrapper import OLT
from nptyping import NDArray


class PolicyFeed(NamedTuple):
    params: chex.ArrayTree
    stacked_frames: chex.ArrayDevice
    legal_actions_mask: types.NestedTensor
    random_key: types.NestedTensor


class PolicyResult(NamedTuple):
    action: chex.ArrayDevice
    extras: Dict[str, chex.ArrayDevice]


class PolicyFn(object):
    def run(self, feed: PolicyFeed):
        raise NotImplementedError


class PriorPolicy(PolicyFn):
    def __init__(
        self, network: mz.nn.NeuralNetwork, epsilon: float, temperature: float
    ) -> None:
        self._network = network
        self._epsilon = epsilon
        self._temperature = temperature

    @partial(jax.jit, static_argnums=(0,))
    @chex.assert_max_traces(n=1)
    def run(self, feed: PolicyFeed) -> PolicyResult:
        network_output = self._network.initial_inference(
            feed.params, add_batch_dim(feed.stacked_frames)
        )
        action_logits = squeeze_batch_dim(network_output.policy_logits)
        chex.assert_rank(action_logits, 1)
        action_entropy = rlax.softmax().entropy(action_logits)
        chex.assert_rank(action_entropy, 0)
        sampler = rlax.epsilon_softmax(self._epsilon, self._temperature).sample
        action = sampler(feed.random_key, action_logits)

        # step_data = mz.logging.JAXBoardStepData({}, {})
        # step_data.add_hk_params(feed.params)
        # step_data.scalars["action_entropy"] = action_entropy
        # step_data.histograms["action_logits"] = action_logits
        return PolicyResult(action=action, extras={})


class RandomPolicy(PolicyFn):
    @partial(jax.jit, static_argnums=(0,))
    @chex.assert_max_traces(n=1)
    def run(self, feed: PolicyFeed) -> PolicyResult:
        action_probs = feed.legal_actions_mask / jnp.sum(feed.legal_actions_mask)
        action_indices = jnp.arange(feed.legal_actions_mask.size)
        action = jax.random.choice(feed.random_key, action_indices, p=action_probs)
        return PolicyResult(action=action, extras={})
