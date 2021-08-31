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


def ucb_score(parent: Node, child: Node):
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


# %%
def get_uuid():
    return uuid.uuid4().hex[:8]


def convert_to_anytree(node: Node, anytree_root=None, action="_"):
    anytree_child = anytree.Node(
        id=get_uuid(),
        name=action,
        parent=anytree_root,
        prior=node.prior,
        reward=np.round(node.reward, 3),
        value=np.round(node.value, 3),
        visits=np.round(node.visit_count, 3),
    )
    for action, child in node.children.items():
        convert_to_anytree(child, anytree_child, action)
    return anytree_child


def nodeattrfunc(node):
    return f'label="V: {node.value:.3f}\nN: {node.visits}"'


def edgeattrfunc(parent, child):
    return f'label="A: {child.name} \np: {child.prior:.3f}\nR: {child.reward:.3f}"'


_partial_exporter = functools.partial(
    anytree.exporter.UniqueDotExporter,
    nodenamefunc=lambda node: node.id,
    nodeattrfunc=nodeattrfunc,
    edgeattrfunc=edgeattrfunc,
)


def anytree_to_png(anytree_root, file_path):
    _partial_exporter(anytree_root).to_picture(file_path)


# %%
safe_epsilon_softmax = jax.jit(rlax.safe_epsilon_softmax(1e-7, 1).probs, backend="cpu")


def expand_node(node: Node, legal_actions_mask, network_output: NeuralNetworkOutput):
    node.hidden_state = network_output.hidden_state
    node.reward = float(network_output.reward)
    action_probs = np.array(safe_epsilon_softmax(network_output.policy_logits))
    action_probs *= legal_actions_mask
    action_probs /= np.sum(action_probs)
    for action, prob in enumerate(action_probs):
        node.children[action] = Node(
            prior=prob,
            parent=node,
            children={},
            value_sum=0.0,
            visit_count=0,
            reward=0.0,
            hidden_state=None,
        )


def select_child(node: Node):
    scores = [
        (ucb_score(node, child), action, child)
        for action, child in node.children.items()
    ]
    _, action, child = max(scores)
    return action, child


def add_exploration_noise(node: Node):
    # TODO: adjustable later
    dirichlet_alpha = 0.25
    frac = 0.25

    actions = list(node.children.keys())
    noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


# %%
def backpropagate(leaf: Node, value: float, discount: float):
    node = leaf
    while True:
        node.value_sum += value
        node.visit_count += 1
        value = node.reward + value * discount
        if node.parent:
            node = node.parent
        else:
            break


# # %%
# anytree_root = convert_to_anytree(node)

# # %%
# print(anytree.RenderTree(anytree_root))
# anytree_to_png(anytree_root, "./policy_tree.png")
# from IPython.display import Image

# Image("./policy_tree.png")


class MonteCarloTreeSearchResult(NamedTuple):
    action_probs: jnp.ndarray


class MonteCarloTreeSearch(object):
    def __init__(
        self, network: NeuralNetwork, num_simulations: int, discount: float
    ) -> None:
        super().__init__()
        # self._network = network
        self._num_simulations = num_simulations
        self._discount = discount
        self._init_inf = jax.jit(network.initial_inference_unbatched, backend="cpu")
        self._recurr_inf = jax.jit(network.recurrent_inference_unbatched, backend="cpu")

    def __call__(self, params, feed: PolicyFeed) -> PolicyResult:
        root_nn_output = self._init_inf(params, feed.stacked_frames)
        root = Node(0)
        expand_node(root, feed.legal_actions_mask, root_nn_output)
        add_exploration_noise(root)

        for _ in range(self._num_simulations):
            self._simulate_once(root, params, feed)
            # action, node = self._select_leaf(root)
            # assert node.parent
            # nn_output = self._recurr_inf(params, node.parent.hidden_state, action)
            # expand_node(node, np.ones_like(feed.legal_actions_mask), nn_output)
            # backpropagate(node, float(nn_output.value), self._discount)

        return PolicyResult(action=jnp.array(0), extras=dict(tree=root))

    def _select_leaf(self, root: Node):
        node = root
        while node.is_expanded:
            action, node = select_child(node)
        return action, node

    def _simulate_once(self, root: Node, params, feed: PolicyFeed):
        action, node = self._select_leaf(root)
        assert node.parent
        nn_output = self._recurr_inf(params, node.parent.hidden_state, action)
        expand_node(node, np.ones_like(feed.legal_actions_mask), nn_output)
        backpropagate(node, float(nn_output.value), self._discount)


# %%
policy_feed = PolicyFeed(
    stacked_frames=np.zeros((1, 6 * 6)),
    legal_actions_mask=np.ones((3,)),
    random_key=jax.random.PRNGKey(0),
)

# %%
num_simulations = 10000
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
mcts = MonteCarloTreeSearch(
    network=network, num_simulations=num_simulations, discount=0.9
)
policy_result = mcts(params, policy_feed)
tree = policy_result.extras["tree"]

# %%
from IPython.display import Image

anytree_root = convert_to_anytree(tree)
print(anytree.RenderTree(anytree_root))
anytree_to_png(anytree_root, "./policy_tree.png")
Image("./policy_tree.png")
