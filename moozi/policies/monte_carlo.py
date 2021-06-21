import chex
import jax
import jax.numpy as jnp
from acme.jax.variable_utils import VariableClient
import numpy as np
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


class MonteCarlo(Policy):
    def __init__(
        self,
        network: NeuralNetwork,
        variable_client: VariableClient,
        num_unroll_steps: int = 5,
        num_simulations_per_action: int = 10,
    ):
        self._network = network
        self._variable_client = variable_client
        self._num_unroll_steps = num_unroll_steps
        self._num_simulations_per_action = num_simulations_per_action

        def _policy_fn(params, feed: PolicyFeed) -> PolicyResult:
            action_space_size = jnp.size(feed.legal_actions_mask)
            starting_actions = jnp.arange(action_space_size).reshape(
                (action_space_size, 1)
            )
            actions_reward_sum = jnp.zeros((action_space_size,))

            starting_hidden_state = network.initial_inference(
                params, add_batch_dim(feed.stacked_frames)
            ).hidden_state

            action_wise_network_output = network.recurrent_inference(
                params, add_batch_dim(starting_hidden_state), starting_actions
            )

            random_following_actions = jax.random.randint(
                feed.random_key,
                (action_space_size, self._num_unroll_steps),
                minval=0,
                maxval=action_space_size,
            )

            def _recurrent_loop_body(i, x):
                network_output, accumulated_rewards = x
                network_output = network.recurrent_inference(
                    params, network_output.hidden_state, random_following_actions[:, i]
                )
                accumulated_rewards = accumulated_rewards + network_output.reward
                return network_output, accumulated_rewards

            accumulated_rewards = jnp.zeros((action_space_size,))
            (_, final_rewards) = jax.lax.fori_loop(
                0,
                self._num_unroll_steps,
                _recurrent_loop_body,
                (action_wise_network_output, accumulated_rewards),
            )
            return PolicyResult(action=jnp.argmax(final_rewards), extras={})
            # actions_reward_sum = actions_reward_sum + final_outputs.reward

            # key, *random_keys = jax.random.split(
            #     feed.random_key, num_simulations_per_action + 1
            # )

            # network.recurrent_inference(params, )

        # random_actions = jax.random.randint(
        #     random_key,
        #     shape=(self._num_unroll_steps - 1,),
        #     minval=0,
        #     maxval=dim_actions,
        # )
        self._policy_fn = _policy_fn

    def run(self, feed: PolicyFeed) -> PolicyResult:
        # dim_actions = jnp.size(feed.legal_actions_mask)
        params = self._variable_client.params
        return self._policy_fn(params, feed)

        # root_hidden_state = self._network.initial_inference(feed.stacked_frames)

        # child_values = jnp.zeros_like(feed.legal_actions_mask)
        # child_hidden_states = params

        # # legal_actions = jnp.where(feed.legal_actions_mask == 1)
        # # values = jnp.zeros(len(legal_actions))
        # # for action in legal_actions:
        # root.children[action] = Node()
        # return PolicyResult(action=action, extras={})

    def update(self, wait: bool = False) -> None:
        self._variable_client.update(wait=wait)
