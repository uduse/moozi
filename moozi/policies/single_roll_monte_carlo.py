from typing import Any, Callable, Dict, NamedTuple
from acme.core_test import StepCountingLearner
import chex
import jax
import jax.numpy as jnp
import numpy as np
import rlax
from acme.jax.utils import add_batch_dim, squeeze_batch_dim
from acme.jax.variable_utils import VariableClient
from jax.ops import index_add, index
from moozi.nn import NeuralNetwork, NeuralNetworkOutput

from .policy import PolicyFn, PolicyFeed, PolicyResult


class Node(NamedTuple):
    network_output: NeuralNetworkOutput
    prior: jnp.ndarray
    children: list


def expand_node(network: NeuralNetwork, params, parent: Node, action_idx: jnp.ndarray):
    child_network_output = network.recurrent_inference_unbatched(
        params,
        parent.network_output.hidden_state,
        action_idx,
    )
    child_prob = rlax.safe_epsilon_softmax(1e-5, 1).probs(
        parent.network_output.policy_logits
    )[action_idx]
    child_node = Node(
        network_output=child_network_output,
        prior=child_prob,
        children=[],
    )
    parent.children.append((action_idx, child_node))
    return child_node


class SingleRollMonteCarloResult(NamedTuple):
    actions_reward_sum: jnp.ndarray
    tree: Node


def make_single_roll_monte_carlo_fn(
    network: NeuralNetwork, num_unroll_steps: int
) -> Callable[[Any, PolicyFeed], SingleRollMonteCarloResult]:
    @jax.jit
    @chex.assert_max_traces(n=1)
    def _policy_fn(params, feed: PolicyFeed) -> SingleRollMonteCarloResult:
        key = feed.random_key
        action_space_size = jnp.size(feed.legal_actions_mask)

        root_network_output = network.initial_inference_unbatched(
            params, feed.stacked_frames
        )

        root_node = Node(
            network_output=root_network_output,
            prior=jnp.array(0),
            children=[],
        )

        actions_reward_sum = jnp.zeros((action_space_size,))
        for child_action_idx in jnp.arange(action_space_size):
            child_node = expand_node(network, params, root_node, child_action_idx)
            actions_reward_sum = index_add(
                actions_reward_sum,
                index[child_action_idx],
                child_node.network_output.reward,
            )

            for _ in jnp.arange(num_unroll_steps):
                key, new_key = jax.random.split(key)
                rollout_action = rlax.safe_epsilon_softmax(1e-7, 1).sample(
                    key=new_key,
                    logits=child_node.network_output.policy_logits,
                )
                child_node = expand_node(network, params, child_node, rollout_action)
                actions_reward_sum = index_add(
                    actions_reward_sum,
                    index[child_action_idx],
                    child_node.network_output.reward,
                )

        return SingleRollMonteCarloResult(actions_reward_sum, root_node)

    return _policy_fn


# def run(self, feed: PolicyFeed) -> PolicyResult:
#     params = self._variable_client.params
#     return self._policy_fn(params, feed)


# def update(self, wait: bool = False) -> None:
#     self._variable_client.update(wait=wait)
