import typing
import math
from collections import namedtuple

import numpy as np

# from moozi import Action, Player, Node, ActionHistory
from moozi.config import Config
from moozi.utils import NetworkOutput, Network

KnownBounds = namedtuple('KnownBounds', ['min', 'max'])


class MinMaxStats(object):
    """
    This is used to compute the normalized Q value (\bar{Q}).

    - A min_max_stats object is instantiated when a search starts.
    - Updated by the backpropagation of each simulation.
    - Affects the ordering of child values during PUCT selection.
    """

    def __init__(self, known_bounds: typing.Optional[KnownBounds] = None):
        self.maximum = known_bounds.max if known_bounds else float('-inf')
        self.minimum = known_bounds.min if known_bounds else float('inf')
    # def __init__(self):
    #     self.maximum = float('-inf')
    #     self.minimum = float('inf')

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values.
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


def ucb_score(config: Config, parent: Node, child: Node, min_max_stats: MinMaxStats) -> float:
    pb_c = config.pb_c_init + \
        math.log((parent.visit_count + config.pb_c_base + 1) / config.pb_c_base)
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
    prior_score = pb_c * child.prior

    if child.visit_count > 0:
        # one-step return as the Q value
        # Q(s, a) = R + \gamma * V(s')
        value_score = child.reward + config.discount * \
            min_max_stats.normalize(child.value)
    else:
        # TODO: when is this used?
        value_score = 0
    return value_score + prior_score


def select_child(
    config: Config, node: Node, min_max_stats: MinMaxStats
) -> typing.Tuple[Action, Node]:
    _, action, child = max(
        (ucb_score(config, node, child, min_max_stats), action,
         child) for action, child in node.children.items())
    return action, child


def expand_node(
    node: Node,
    to_play: Player,
    actions: typing.List[Action],
    network_output: NetworkOutput
):
    node.to_play = to_play
    node.hidden_state = network_output.hidden_state
    node.reward = network_output.reward

    action_probs = np.exp([network_output.policy_logits[a] for a in actions])
    probs_sum = np.sum(action_probs)
    for action in actions:
        child_prior_prob = action_probs[action]
        child = Node(child_prior_prob / probs_sum)
        node.children[action] = child


def backpropagate(
    search_path: typing.List[Node],
    value: float,
    to_play: Player,
    discount: float,
    min_max_stats: MinMaxStats
):
    for node in reversed(search_path):
        if node.to_play == to_play:
            node.value_sum += value
        else:
            node.value_sum += -value
        min_max_stats.update(node.value)

        node.visit_count += 1
        value = node.reward + discount * value


def add_exploration_noise(config: Config, node: Node):
    actions = list(node.children.keys())
    noise = np.random.dirichlet(
        [config.root_dirichlet_alpha] * len(actions))
    frac = config.root_exploration_fraction
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


def run_mcts(
    config: Config,
    root: Node,
    action_history: ActionHistory,
    network: Network
):
    """
    We need two things to keep track of MCTS search status:
        - a game history that records all the actions have taken so far
        - a search history that records all searched nodes
    """
    min_max_stats = MinMaxStats()

    for _ in range(config.num_simulations):
        action_history = action_history.clone()
        node = root
        search_path = [node]

        while node.expanded():
            action, node = select_child(config, node, min_max_stats)
            action_history.add_action(action)
            search_path.append(node)

        parent = search_path[-2]
        network_output = network.recurrent_inference(
            parent.hidden_state, action_history.last_action())
        expand_node(node, action_history.to_play(),
                    action_history.action_space(), network_output)
        backpropagate(search_path, network_output.value,
                      action_history.to_play(), config.discount, min_max_stats)
