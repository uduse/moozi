import chex
import jax
import jax.numpy as jnp
import numpy as np
import rlax
from acme.jax.utils import add_batch_dim, squeeze_batch_dim
from acme.jax.variable_utils import VariableClient
from jax.ops import index_add, index
from moozi.nerual_network import NeuralNetwork, NeuralNetworkOutput

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


class MonteCarlo(Policy):
    def __init__(
        self,
        network: NeuralNetwork,
        variable_client: VariableClient,
        num_unroll_steps: int = 5,
        num_simulations_per_action: int = 10,
        epsilon: float = 0.1,
    ):
        @jax.jit
        @chex.assert_max_traces(n=1)
        def _policy_fn(params, feed: PolicyFeed) -> PolicyResult:
            action_space_size = jnp.size(feed.legal_actions_mask)
            actions_reward_sum = jnp.zeros((action_space_size,))

            starting_hidden_state = network.initial_inference(
                params, add_batch_dim(feed.stacked_frames)
            ).hidden_state
            starting_hidden_state = squeeze_batch_dim(starting_hidden_state)

            random_actions_for_unrolling = jax.random.randint(
                feed.random_key,
                (action_space_size, num_simulations_per_action, num_unroll_steps),
                minval=jnp.array(0),
                maxval=jnp.array(action_space_size),
            )

            for child_action_idx in jnp.arange(action_space_size):
                for sim_idx in range(num_simulations_per_action):
                    hidden_state = starting_hidden_state
                    child_network_output = network.recurrent_inference(
                        params,
                        add_batch_dim(hidden_state),
                        add_batch_dim(child_action_idx),
                    )
                    actions_reward_sum = index_add(
                        actions_reward_sum,
                        index[child_action_idx],
                        child_network_output.reward.item(),
                    )
                    roll_out_hidden_state = child_network_output.hidden_state
                    for step_idx in range(num_unroll_steps):
                        random_action_for_unroll = random_actions_for_unrolling[
                            child_action_idx, sim_idx, step_idx
                        ]
                        network_output = network.recurrent_inference(
                            params,
                            add_batch_dim(roll_out_hidden_state),
                            add_batch_dim(random_action_for_unroll),
                        )
                        actions_reward_sum = index_add(
                            actions_reward_sum,
                            index[child_action_idx],
                            network_output.reward,
                        )
                        roll_out_hidden_state = network_output.hidden_state

            child_action_idx = rlax.epsilon_greedy(epsilon).sample(
                feed.random_key, actions_reward_sum
            )

            return PolicyResult(action=child_action_idx, extras={})

        self._policy_fn = _policy_fn
        self._variable_client = variable_client

    def run(self, feed: PolicyFeed) -> PolicyResult:
        # dim_actions = jnp.size(feed.legal_actions_mask)
        params = self._variable_client.params
        return self._policy_fn(params, feed)

    def update(self, wait: bool = False) -> None:
        self._variable_client.update(wait=wait)
