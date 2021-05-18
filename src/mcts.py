import typing
import math
from dataclasses import dataclass
from collections import namedtuple

import numpy as np

import mozi as mz
# from mozi.utils import NetworkOutput
# from mozi.config import Config

Action = typing.NewType('Action', int)
Player = typing.NewType('Player', int)

KnownBounds = namedtuple('KnownBounds', ['min', 'max'])


class MinMaxStats(object):
    def __init__(self, known_bounds: typing.Optional[KnownBounds]):
        self.maximum = known_bounds.max if known_bounds else float('-inf')
        self.minimum = known_bounds.min if known_bounds else float('inf')

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values.
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class Node(object):
    def __init__(self, prior: float):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children: typing.List[Node] = {}
        self.hidden_state = None
        self.reward = 0

    def expanded(self) -> bool:
        return len(self.children) > 0

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


# @dataclass
# class Config(object):
#     num_simulations: int = 25
#     pb_c_init: float = 1.25
#     pb_c_base: float = 19652

#     dim: int = 5


def _initial_reference(image):
    return NetworkOutput()


def run_mcts(
    config: MuZeroConfig,
    root: Node,
    action_history: ActionHistory,
    network: Network
):
    for _ in range(config.num_simulations):
        node = root


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
    # config: Config,
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
    discount: float,
    to_play: Player,
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


def add_exploration_noise(config: MuZeroConfig, node: Node):
    actions = list(node.children.keys())
    noise = np.random.dirichlet(
        [config.root_dirichlet_alpha] * len(actions))
    frac = config.root_exploration_fraction
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


def run_mcts(config: Config, root: Node):
    for _ in range(config.num_simulations):
        node = root



def main():
    # node = Node(1)
    # mcts = MonteCarloTreeSearch(config)
    # print(mcts)
    config = Config()
    # select_child()


if __name__ == '__main__':
    # main()
