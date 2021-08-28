import copy
import chex
import math
from typing import Any, Dict, NamedTuple, Optional

import jax.numpy as jnp
import numpy as np
from moozi.nn import NeuralNetwork, NeuralNetworkOutput

from .policy import PolicyFeed, PolicyFn, PolicyResult

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


@chex.dataclass
class Node:
    prior: float
    parent: Any  # Optional[Node]
    children: dict
    value_sum: float
    visit_count: int
    last_reward: float
    hidden_state: jnp.ndarray


def get_value(node: Node) -> float:
    return node.value_sum / node.visit_count


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


# def ucb_score(
#     config: Config, parent: Node, child: Node, min_max_stats: MinMaxStats
# ) -> float:
#     pb_c = config.pb_c_init + math.log(
#         (parent.visit_count + config.pb_c_base + 1) / config.pb_c_base
#     )
#     pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
#     prior_score = pb_c * child.prior

#     if child.visit_count > 0:
#         # one-step return as the Q value
#         # Q(s, a) = R + \gamma * V(s')
#         value_score = child.reward + config.discount * min_max_stats.normalize(
#             child.value
#         )
#     else:
#         # TODO: when is this used?
#         value_score = 0
#     return value_score + prior_score


# def select_child(
#     config: Config, node: Node, min_max_stats: MinMaxStats
# ) -> typing.Tuple[Action, Node]:
#     _, action, child = max(
#         (ucb_score(config, node, child, min_max_stats), action, child)
#         for action, child in node.children.items()
#     )
#     return action, child


# def expand_node(
#     node: Node,
#     to_play: Player,
#     actions: typing.List[Action],
#     network_output: NetworkOutput,
# ):
#     node.to_play = to_play
#     node.hidden_state = network_output.hidden_state
#     node.reward = network_output.reward

#     action_probs = np.exp([network_output.policy_logits[a] for a in actions])
#     probs_sum = np.sum(action_probs)
#     for action in actions:
#         child_prior_prob = action_probs[action]
#         child = Node(child_prior_prob / probs_sum)
#         node.children[action] = child


# def backpropagate(
#     search_path: typing.List[Node],
#     value: float,
#     to_play: Player,
#     discount: float,
#     min_max_stats: MinMaxStats,
# ):
#     for node in reversed(search_path):
#         if node.to_play == to_play:
#             node.value_sum += value
#         else:
#             node.value_sum += -value
#         min_max_stats.update(node.value)

#         node.visit_count += 1
#         value = node.reward + discount * value


# def add_exploration_noise(config: Config, node: Node):
#     actions = list(node.children.keys())
#     noise = np.random.dirichlet([config.root_dirichlet_alpha] * len(actions))
#     frac = config.root_exploration_fraction
#     for a, n in zip(actions, noise):
#         node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


# def run_mcts(
#     config: Config, root: Node, action_history: ActionHistory, network: Network
# ):
#     """
#     We need two things to keep track of MCTS search status:
#         - a game history that records all the actions have taken so far
#         - a search history that records all searched nodes
#     """
#     min_max_stats = MinMaxStats()

#     for _ in range(config.num_simulations):
#         action_history = action_history.clone()
#         node = root
#         search_path = [node]

#         while node.expanded():
#             action, node = select_child(config, node, min_max_stats)
#             action_history.add_action(action)
#             search_path.append(node)

#         parent = search_path[-2]
#         network_output = network.recurrent_inference(
#             parent.hidden_state, action_history.last_action()
#         )
#         expand_node(
#             node,
#             action_history.to_play(),
#             action_history.action_space(),
#             network_output,
#         )
#         backpropagate(
#             search_path,
#             network_output.value,
#             action_history.to_play(),
#             config.discount,
#             min_max_stats,
#         )
