# %%
import copy
import chex
import math
from typing import Any, Dict, NamedTuple, Optional

import jax.numpy as jnp
import numpy as np
from moozi.nn import NeuralNetwork, NeuralNetworkOutput

from moozi.policies.policy import PolicyFeed, PolicyFn, PolicyResult

# root = Node(0)
# current_observation = game.make_image(-1)
# expand_node(
#     root,
#     game.to_play(),
#     game.legal_actions(),
#     network.initial_inference(current_observation),
# )
# add_exploration_noise(config, root)

# # We then run a Monte Carlo Tree Search using only action sequences and the
# # model learned by the network.
#     run_mcts(config, root, game.action_history(), network)


# %%
@chex.dataclass
class Node:
    prior: float
    parent: Any  # Optional[Node]
    children: dict
    value_sum: float
    visit_count: int
    last_reward: float
    hidden_state: jnp.ndarray

    @property
    def value(self) -> float:
        assert self.visit_count > 0
        return self.value_sum / self.visit_count


def ucb_score(parent: Node, child: Node):
    pb_c_base = 19652.0
    pb_c_init = 1.25
    # TODO: obviously this should be a parameter
    discount = 0.99

    pb_c = jnp.log((parent.visit_count + pb_c_base + 1) / pb_c_base) + pb_c_init
    pb_c *= jnp.sqrt(parent.visit_count) / (child.visit_count + 1)
    prior_score = pb_c * child.prior

    if child.visit_count > 0:
        value_score = child.last_reward + discount * get_value(child)
    else:
        value_score = 0.0
    return prior_score + value_score


# %%
node = Node(
    prior=0.0,
    parent=None,
    children={},
    value_sum=0.0,
    visit_count=0,
    last_reward=0.0,
    hidden_state=np.array(0),
)
# %%


def expand_node(node: Node):
    pass


def select_child(node: Node):
    pass


def is_expanded(node: Node):
    return node.visit_count > 0


# def copy_node(node: Node) -> Node:
#     return Node(
#         prior=copy.deepcopy(node.prior),
#         parent=copy.deepcopy(node.parent),
#         children=copy.deepcopy(node.children),
#         value_sum=copy.deepcopy(node.value_sum),
#         visit_count=copy.deepcopy(node.visit_count),
#         last_reward=copy.deepcopy(node.last_reward),
#         hidden_state=copy.deepcopy(node.hidden_state),
#     )


def backpropagate(leaf: Node, value: float, discount: float):
    node = leaf
    while node.parent:
        node = node.parent
        node.value_sum += value
        node.visit_count += 1
        value = node.last_reward + value * discount


def make_root_node(hidden_state: jnp.ndarray) -> Node:
    return Node(
        prior=0.0,
        parent=None,
        children={},
        value_sum=0.0,
        last_reward=0.0,
        visit_count=0,
        hidden_state=hidden_state,
    )


class MonteCarloTreeSearchResult(NamedTuple):
    action_probs: jnp.ndarray


class MonteCarloTreeSearch(object):
    def __init__(self, network: NeuralNetwork, num_simulations: int) -> None:
        super().__init__()
        self._network = network
        self._num_simulations = num_simulations

    def __call__(self, params, feed: PolicyFeed) -> PolicyResult:
        root_nn_output = self._network.initial_inference_unbatched(
            params, feed.stacked_frames
        )
        root = make_root_node(root_nn_output.hidden_state)
        for _ in range(self._num_simulations):
            node = root
            while is_expanded(node):
                node = select_child(node)
            expand_node(node)
            backpropagate()

        return PolicyResult(action=jnp.array(0), extras={})
