import chex
import jax
import jax.numpy as jnp
from acme.jax.variable_utils import VariableClient
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


def _expand_node(node, actions):
    pass


class MonteCarlo(Policy):
    def __init__(
        self,
        network: NeuralNetwork,
        variable_client: VariableClient,
        num_unroll_steps: int = 5,
        num_simulations_per_action: int = 50,
    ):
        self._network = network
        self._variable_client = variable_client
        self._num_unroll_steps = num_unroll_steps
        self._num_simulations_per_action = num_simulations_per_action

    def run(self, feed: PolicyFeed) -> PolicyResult:
        dim_actions = jnp.size(feed.legal_actions_mask)
        params = self._variable_client.params
        root_hidden_state = self._network.initial_inference(feed.stacked_frames)

        @jax.jit
        @chex.assert_max_traces(n=1)
        def _simulate(random_key, network_output: NeuralNetworkOutput):
            random_actions = jax.random.randint(
                random_key,
                shape=(self._num_unroll_steps - 1,),
                minval=0,
                maxval=dim_actions,
            )

            def _loop_body(i, x):
                return self._network.recurrent_inference(x, random_actions[i])

            network_output = jax.lax.fori_loop(
                0, self._num_simulations_per_action, _loop_body, network_output
            )
            return network_output.value

        child_values = jnp.zeros_like(feed.legal_actions_mask)
        child_hidden_states = params

        # legal_actions = jnp.where(feed.legal_actions_mask == 1)
        # values = jnp.zeros(len(legal_actions))
        # for action in legal_actions:
        #     root.children[action] = Node()
        return PolicyResult(action=action, extras={})

    def update(self, wait: bool = False) -> None:
        self._variable_client.update(wait=wait)
