# %%
import copy
import functools
import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, NamedTuple, Optional

import anytree
import chex
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

    def expand_node(self, legal_actions_mask, network_output: NeuralNetworkOutput):
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
        dirichlet_alpha = 0.25
        frac = 0.25

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

    def get_children_visit_counts(self):
        visit_counts = np.zeros_like(self._all_actions_mask)
        for action, child in self.children.items():
            visit_counts[action] = child.visit_count
        return visit_counts


# %%
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


# %%


class MonteCarloTreeSearchResult(NamedTuple):
    visit_counts: jnp.ndarray
    tree: Node


class MonteCarloTreeSearch(object):
    def __init__(
        self,
        network: NeuralNetwork,
        num_simulations: int,
        discount: float,
        dim_actions: int,
    ) -> None:
        super().__init__()
        # self._network = network
        self._num_simulations = num_simulations
        self._discount = discount
        self._init_inf = jax.jit(network.initial_inference_unbatched, backend="cpu")
        self._recurr_inf = jax.jit(network.recurrent_inference_unbatched, backend="cpu")
        self._all_actions_mask = np.ones((dim_actions,), dtype=np.int32)

    def __call__(self, params, feed: PolicyFeed) -> MonteCarloTreeSearchResult:
        root = self.get_root(feed)

        for _ in range(self._num_simulations):
            self.simulate(root, params)

        visit_counts = root.get_children_visit_counts()
        return MonteCarloTreeSearchResult(visit_counts=visit_counts, tree=root)

    def get_root(self, feed: PolicyFeed) -> Node:
        root_nn_output = self._init_inf(params, feed.stacked_frames)
        root = Node(0)
        root.expand_node(feed.legal_actions_mask, root_nn_output)
        root.add_exploration_noise()
        return root

    def select_leaf(self, root: Node):
        node = root
        while node.is_expanded:
            action, node = node.select_child()
        return action, node

    def simulate(
        self,
        root: Node,
        params,
    ):
        action, node = self.select_leaf(root)
        assert node.parent
        # nn_output = self._recurr_inf(params, node.parent.hidden_state, action)
        nn_output = self.recurr_inf(params, node, action)
        node.expand_node(self._all_actions_mask, nn_output)
        node.backpropagate(float(nn_output.value), self._discount)

    def recurr_inf(self, params, node, action):
        return self._recurr_inf(params, node.parent.hidden_state, action)


# %%
policy_feed = PolicyFeed(
    stacked_frames=np.zeros((1, 6 * 6)),
    legal_actions_mask=np.ones((3,)),
    random_key=jax.random.PRNGKey(0),
)

# %%
nn_spec = NeuralNetworkSpec(
    stacked_frames_shape=(1, 6 * 6),
    dim_repr=64,
    dim_action=3,
    repr_net_sizes=(128, 128),
    pred_net_sizes=(128, 128),
    dyna_net_sizes=(128, 128),
)
network = get_network(nn_spec)
params = network.init(jax.random.PRNGKey(0))

# %%
# %%time
num_simulations = 800
mcts = MonteCarloTreeSearch(
    network=network, num_simulations=num_simulations, discount=0.9, dim_actions=3
)
mcts_result = mcts(params, policy_feed)
tree = mcts_result.tree

# %%
from IPython.display import Image

anytree_root = convert_to_anytree(tree)
# print(anytree.RenderTree(anytree_root))
anytree_to_png(anytree_root, "./policy_tree.png")
# anytree_to_json(anytree_root, "./policy_tree.json")
Image("./policy_tree.png")
