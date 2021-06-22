import chex
import jax
import jax.numpy as jnp
from acme.jax.variable_utils import VariableClient
import numpy as np
import rlax
from moozi.nerual_network import NeuralNetwork, NeuralNetworkOutput
from acme.jax.utils import add_batch_dim, squeeze_batch_dim

from .policy import Policy, PolicyFeed, PolicyResult

# class Node(object):
#     def __init__(self):
#         self.value_sum: float = 0
#         self.children: dict = {}
#         self.hidden_state = None
#         self.visit_count: int = 0

#     def is_expanded(self) -> bool:
#         return len(self.children) > 0

#     # def add(self, value):
#     #     self.value_sum += value
#     #     self.visit_count += 1

#     @property
#     def value(self) -> float:
#         if self.visit_count == 0:
#             return 0
#         return self.value_sum / self.visit_count


class SingleRollMonteCarlo(Policy):
    def __init__(
        self,
        network: NeuralNetwork,
        variable_client: VariableClient,
        num_unroll_steps: int = 5,
        epsilon: float = 0.1,
    ):
        @jax.jit
        @chex.assert_max_traces(n=1)
        def _policy_fn(params, feed: PolicyFeed) -> PolicyResult:
            action_space_size = jnp.size(feed.legal_actions_mask)
            starting_actions = jnp.arange(action_space_size)
            actions_reward_sum = jnp.zeros((action_space_size,))

            starting_hidden_state = network.initial_inference(
                params, add_batch_dim(feed.stacked_frames)
            ).hidden_state
            starting_hidden_state = squeeze_batch_dim(starting_hidden_state)

            starting_hidden_state = jnp.repeat(
                starting_hidden_state[jnp.newaxis, :], action_space_size, axis=0
            )

            action_wise_network_output = network.recurrent_inference(
                params, starting_hidden_state, starting_actions
            )

            random_following_actions = jax.random.randint(
                feed.random_key,
                (action_space_size, num_unroll_steps),
                minval=jnp.array(0),
                maxval=jnp.array(action_space_size),
            )

            def _all_actions_one_simulation_loop_body(i, x):
                network_output, accumulated_rewards = x
                network_output = network.recurrent_inference(
                    params, network_output.hidden_state, random_following_actions[:, i]
                )
                accumulated_rewards = accumulated_rewards + network_output.reward
                return network_output, accumulated_rewards

            accumulated_rewards = jnp.zeros((action_space_size,))
            (_, accumulated_rewards) = jax.lax.fori_loop(
                0,
                num_unroll_steps,
                _all_actions_one_simulation_loop_body,
                (action_wise_network_output, accumulated_rewards),
            )

            actions_reward_sum = actions_reward_sum + accumulated_rewards
            action = rlax.epsilon_greedy(epsilon).sample(
                feed.random_key, actions_reward_sum
            )

            return PolicyResult(action=action, extras={})

        self._policy_fn = _policy_fn
        self._variable_client = variable_client

    def run(self, feed: PolicyFeed) -> PolicyResult:
        # dim_actions = jnp.size(feed.legal_actions_mask)
        params = self._variable_client.params
        return self._policy_fn(params, feed)

    def update(self, wait: bool = False) -> None:
        self._variable_client.update(wait=wait)
