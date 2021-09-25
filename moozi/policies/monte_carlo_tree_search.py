import functools
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, NamedTuple, Optional

import anytree
import jax
import jax.numpy as jnp
import numpy as np
import rlax
from moozi.nn import NeuralNetwork, NeuralNetworkOutput, NeuralNetworkSpec, get_network
from moozi.policies.policy import PolicyFeed, PolicyFn, PolicyResult


# %%
_safe_epsilon_softmax = jax.jit(rlax.safe_epsilon_softmax(1e-7, 1).probs, backend="cpu")


@dataclass
class Node(object):
    prior: float
    player: int = 0
    parent: Optional["Node"] = None
    children: dict = field(default_factory=dict)
    value_sum: float = 0.0
    visit_count: int = 0
    reward: float = 0.0
    hidden_state: Optional[np.ndarray] = None

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        else:
            return self.value_sum / self.visit_count

    @property
    def is_expanded(self) -> bool:
        return len(self.children) > 0

    def expand_node(self, network_output: NeuralNetworkOutput, legal_actions_mask):
        self.hidden_state = network_output.hidden_state
        self.reward = float(network_output.reward)
        action_probs = np.array(_safe_epsilon_softmax(network_output.policy_logits))
        action_probs *= legal_actions_mask
        action_probs /= np.sum(action_probs)
        for action, prob in enumerate(action_probs):
            self.children[action] = Node(
                prior=prob,
                parent=self,
                children={},
                value_sum=0.0,
                visit_count=0,
                reward=0.0,
                hidden_state=None,
            )

    def select_child(self):
        scores = [
            (Node.ucb_score(parent=self, child=child), action, child)
            for action, child in self.children.items()
        ]
        _, action, child = max(scores)
        return action, child

    def add_exploration_noise(self):
        # TODO: adjustable later
        dirichlet_alpha = 0.2
        frac = 0.2

        actions = list(self.children.keys())
        noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
        for a, n in zip(actions, noise):
            self.children[a].prior = self.children[a].prior * (1 - frac) + n * frac

    def backpropagate(self, value: float, discount: float):
        node = self
        while True:
            node.value_sum += value
            node.visit_count += 1
            value = node.reward + value * discount
            if node.parent:
                node = node.parent
            else:
                break

    def get_children_visit_counts(self):
        visit_counts = {}
        for action, child in self.children.items():
            visit_counts[action] = child.visit_count
        return visit_counts

    def get_children_values(self):
        values = {}
        for action, child in self.children.items():
            values[action] = child.value
        return values

    def select_leaf(self):
        node = self
        while node.is_expanded:
            action, node = node.select_child()
        return action, node

    @classmethod
    def ucb_score(cls, parent: "Node", child: "Node"):
        pb_c_base = 19652.0
        pb_c_init = 1.25
        # TODO: obviously this should be a parameter
        discount = 0.99

        pb_c = np.log((parent.visit_count + pb_c_base + 1) / pb_c_base) + pb_c_init
        pb_c *= np.sqrt(parent.visit_count) / (child.visit_count + 1)
        prior_score = pb_c * child.prior

        if child.visit_count > 0:
            value_score = child.reward + discount * child.value
        else:
            value_score = 0.0
        return prior_score + value_score


def get_uuid():
    return uuid.uuid4().hex[:8]


def convert_to_anytree(node: Node, anytree_node=None, action="_"):
    if node.visit_count > 0:
        anytree_node = anytree.Node(
            id=get_uuid(),
            name=action,
            parent=anytree_node,
            prior=str(np.round(node.prior, 3)),
            reward=str(np.round(node.reward, 3)),
            value=str(np.round(node.value, 3)),
            visits=str(np.round(node.visit_count, 3)),
        )
        for action, child in node.children.items():
            convert_to_anytree(child, anytree_node, action)
        return anytree_node
    else:
        return anytree_node


def nodeattrfunc(node):
    return f'label="V: {node.value}\nN: {node.visits}"'


def edgeattrfunc(parent, child):
    return f'label="A: {child.name} \np: {child.prior}\nR: {child.reward}"'


_partial_dot_exporter = functools.partial(
    anytree.exporter.UniqueDotExporter,
    nodenamefunc=lambda node: node.id,
    nodeattrfunc=nodeattrfunc,
    edgeattrfunc=edgeattrfunc,
)


def anytree_to_png(anytree_root, file_path):
    _partial_dot_exporter(anytree_root).to_picture(file_path)


def anytree_to_json(anytree_root, file_path):
    json_s = anytree.exporter.JsonExporter(indent=2, sort_keys=True).export(
        anytree_root
    )
    with open(file_path, "w") as f:
        f.write(json_s)


class MonteCarloTreeSearch(object):
    def __init__(
        self,
        network: NeuralNetwork,
        num_simulations: int,
        discount: float,
        dim_action: int,
    ) -> None:
        super().__init__()
        # self._network = network
        self._num_simulations = num_simulations
        self._discount = discount
        self._init_inf = jax.jit(network.initial_inference_unbatched, backend="cpu")
        self._recurr_inf = jax.jit(network.recurrent_inference_unbatched, backend="cpu")
        self._all_actions_mask = np.ones((dim_action,), dtype=np.int32)

    def __call__(self, params, feed: PolicyFeed) -> Node:
        root = self.get_root(params, feed)

        for _ in range(self._num_simulations):
            self.simulate_once(params, root)

        return root

    def get_root(self, params, feed: PolicyFeed) -> Node:
        root_nn_output = self._init_inf(params, feed.stacked_frames)
        root = Node(0)
        root.expand_node(root_nn_output, feed.legal_actions_mask)
        root.add_exploration_noise()
        return root

    def simulate_once(
        self,
        params,
        root: Node,
    ):
        action, node = root.select_leaf()
        assert node.parent
        nn_output = self._recurr_inf(params, node.parent.hidden_state, action)
        node.expand_node(nn_output, self._all_actions_mask)
        node.backpropagate(float(nn_output.value), self._discount)

    @property
    def all_actions_mask(self):
        return self._all_actions_mask


def train_policy_fn(
    mcts: MonteCarloTreeSearch, params, feed: PolicyFeed
) -> PolicyResult:
    mcts_tree = mcts(params, feed)

    action_probs = np.zeros_like(mcts.all_actions_mask, dtype=np.float32)
    for a, visit_count in mcts_tree.get_children_visit_counts().items():
        action_probs[a] = visit_count
    action_probs /= np.sum(action_probs)

    action, _ = mcts_tree.select_child()
    return PolicyResult(
        action=np.array(action, dtype=np.int32),
        extras={"tree": mcts_tree, "action_probs": action_probs},
    )


def eval_policy_fn(
    mcts: MonteCarloTreeSearch, params, feed: PolicyFeed
) -> PolicyResult:
    policy_result = train_policy_fn(mcts, params, feed)

    mcts_tree = policy_result.extras["tree"]

    child_values = np.zeros_like(mcts.all_actions_mask, dtype=np.float32)
    for action, value in mcts_tree.get_children_values().items():
        child_values[action] = value

    policy_result.extras.update(
        {
            "child_values": child_values,
        }
    )
    return policy_result
