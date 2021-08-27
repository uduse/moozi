from collections import defaultdict
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union

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
    stacked_frames: jnp.ndarray
    legal_actions_mask: jnp.ndarray
    random_key: jnp.ndarray


class PolicyResult(NamedTuple):
    action: jnp.ndarray
    extras: Dict[str, Any]


# class ActionSelector(object):
#     pass


# def epsilon_greedy(
#     random_key, epsilon, preferences, legal_actions_mask
# ) -> Tuple[jnp.ndarray, jnp.ndarray]:
#     action_probs = rlax.epsilon_greedy(epsilon).probs(preferences)
#     legal_action_probs = action_probs * legal_actions_mask
#     action = rlax.categorical_sample(random_key, legal_action_probs)
#     return (action, legal_action_probs)


PolicyFn = Callable[[Any, PolicyFeed], PolicyResult]

# TODO: make this callable, not a function
# class PriorPolicy(PolicyFn):
#     def __init__(
#         self,
#         network: mz.nn.NeuralNetwork,
#         variable_client: VariableClient,
#         epsilon: float = 0.05,
#         temperature: float = 1.0,
#     ) -> None:
#         @jax.jit
#         @chex.assert_max_traces(n=1)
#         def _policy_fn(params, stacked_frames, random_key):
#             network_output = network.initial_inference(
#                 params, add_batch_dim(stacked_frames)
#             )
#             action_logits = squeeze_batch_dim(network_output.policy_logits)
#             chex.assert_rank(action_logits, 1)
#             sampler = rlax.epsilon_softmax(epsilon, temperature).sample
#             action = sampler(random_key, action_logits)

#             # action_entropy = rlax.softmax().entropy(action_logits)
#             # chex.assert_rank(action_entropy, 0)
#             # step_data = mz.logging.JAXBoardStepData({}, {})
#             # step_data.add_hk_params(feed.params)
#             # step_data.scalars["action_entropy"] = action_entropy
#             # step_data.histograms["action_logits"] = action_logits
#             return PolicyResult(action_probs=action, extras={})

#         self._policy_fn = _policy_fn
#         self._variable_client = variable_client

#     def run(self, feed: PolicyFeed) -> PolicyResult:
#         params = self._variable_client.params
#         return self._policy_fn(params, feed.stacked_frames, feed.random_key)

#     def update(self, wait: bool = False) -> None:
#         self._variable_client.update(wait)


# class RandomPolicy(PolicyFn):
#     @partial(jax.jit, static_argnums=(0,))
#     @chex.assert_max_traces(n=1)
#     def run(self, feed: PolicyFeed) -> PolicyResult:
#         action_probs = feed.legal_actions_mask / jnp.sum(feed.legal_actions_mask)
#         action_indices = jnp.arange(feed.legal_actions_mask.size)
#         action = jax.random.choice(feed.random_key, action_indices, p=action_probs)
#         return PolicyResult(action_probs=action, extras={})
