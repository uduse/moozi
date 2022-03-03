import functools
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
from acme.jax.utils import to_numpy

import anytree
import jax
import jax.numpy as jnp
import numpy as np
from moozi.core import BASE_PLAYER


class SearchStrategy(Enum):
    # default MuZero strategy
    TWO_PLAYER = auto()

    # VQVAE + MuZero
    ONE_PLAYER = auto()

    # TODO: VQVAE + MuZero
    VQ_HYBRID = auto()
    VQ_PURE = auto()
    VQ_JUMPY = auto()


def get_next_player(strategy: SearchStrategy, to_play: int) -> int:
    if strategy == SearchStrategy.TWO_PLAYER:
        if to_play == 0:
            return 1
        elif to_play == 1:
            return 0
        else:
            raise ValueError(f"Invalid to_play: {to_play}")

    elif strategy == SearchStrategy.ONE_PLAYER:
        return 0

    else:
        raise NotImplementedError(f"{strategy} not implemented")


def get_prev_player(strategy: SearchStrategy, to_play: int) -> int:
    if strategy == SearchStrategy.TWO_PLAYER:
        if to_play == 0:
            return 1
        elif to_play == 1:
            return 0
        else:
            raise ValueError(f"Invalid to_play: {to_play}")

    elif strategy == SearchStrategy.ONE_PLAYER:
        return 0

    else:
        raise NotImplementedError(f"{strategy} not implemented")


# TODO: move to NN related file
def reorient(item: float, player: int) -> float:
    """Reorient value or reward to the root_player's perspective."""
    if player == BASE_PLAYER:
        return item
    else:
        return -item


# TODO: findout the most efficient version of softmax implementation
# _safe_epsilon_softmax = jax.jit(rlax.safe_epsilon_softmax(1e-7, 1).probs, backend="cpu")
# softmax = jax.jit(jax.nn.softmax, backend="cpu")
@functools.partial(jax.jit, backend="cpu")
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = jnp.exp(x - jnp.max(x))
    return e_x / jnp.sum(e_x)


@dataclass
class Node(object):
    prior: float
    player: int = 0
    parent: Optional["Node"] = None
    name: str = ""
    children: dict = field(default_factory=dict)
    value_sum: float = 0.0
    visit_count: int = 0
    last_reward: float = 0.0
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

    def expand_node(
        self,
        hidden_state: np.ndarray,
        reward: float,
        policy_logits: np.ndarray,
        legal_actions_mask: np.ndarray,
        next_player: int,
    ):

        self.hidden_state = hidden_state
        self.last_reward = reward
        action_probs = self._compute_action_probs(policy_logits, legal_actions_mask)

        for action, prob in enumerate(action_probs):
            if prob > 0.0:
                self.children[action] = Node(
                    prior=prob,
                    player=next_player,
                    parent=self,
                    name=self.name + f"_{action}",
                    children={},
                    value_sum=0.0,
                    visit_count=0,
                    last_reward=0.0,
                    hidden_state=None,
                )

    def _compute_action_probs(self, policy_logits, legal_actions_mask):
        action_probs = np.array(softmax(policy_logits))
        action_probs *= legal_actions_mask
        action_probs /= np.sum(action_probs)
        return action_probs

    def select_child(self, discount: float):
        scores = []
        for action, child in self.children.items():
            ucb_score = Node.ucb_score(parent=self, child=child, discount=discount)
            scores.append((ucb_score, action, child))

        # TODO: break ties randomly?
        _, action, child = max(scores)
        return action, child

    def add_exploration_noise(self, dirichlet_alpha: float = 0.2, frac: float = 0.2):
        actions = list(self.children.keys())
        noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
        for a, n in zip(actions, noise):
            self.children[a].prior = self.children[a].prior * (1 - frac) + n * frac

    def backpropagate(self, value: float, discount: float):
        """[summary]

        [extended_summary]

        :param value: value based on the BASE_PLAYER's perspective
        :type value: float
        :param discount: [description]
        :type discount: float
        """
        node = self
        while True:
            node.value_sum += value
            node.visit_count += 1

            if node.parent:
                value = node.last_reward + value * discount
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

    def get_children_visit_counts_as_probs(self, dim_action):
        action_probs = np.zeros((dim_action,), dtype=np.float32)
        for a, visit_count in self.get_children_visit_counts().items():
            action_probs[a] = visit_count
        action_probs /= np.sum(action_probs)
        return action_probs

    def select_leaf(self, discount: float):
        node = self
        while node.is_expanded:
            action, node = node.select_child(discount=discount)
        return action, node

    @staticmethod
    def ucb_score(parent: "Node", child: "Node", discount: float = 1.0) -> float:
        pb_c_base = 19652.0
        pb_c_init = 1.25
        # TODO: obviously this `discount` should be a parameter

        pb_c = np.log((parent.visit_count + pb_c_base + 1) / pb_c_base) + pb_c_init
        pb_c *= np.sqrt(parent.visit_count) / (child.visit_count + 1)
        prior_score = pb_c * child.prior

        if child.visit_count > 0:
            value_score = child.last_reward + discount * child.value
            # max(-scores) if the parent player is not the BASE_PLAYER
            if parent.player != BASE_PLAYER:
                value_score = -value_score
        else:
            value_score = 0.0

        return prior_score + value_score


def get_uuid():
    return uuid.uuid4().hex[:8]


def convert_to_anytree(node: Node, anytree_node=None, last_action: int = 0):
    anytree_node = anytree.Node(
        id=get_uuid(),
        name=node.name,
        last_action=last_action,
        parent=anytree_node,
        to_play=node.player,
        prior=np.round(node.prior, 3),
        reward=np.round(node.last_reward, 3),
        value=np.round(node.value, 3),
        visits=node.visit_count,
    )
    for action, child in node.children.items():
        convert_to_anytree(child, anytree_node, action)
    return anytree_node


def nodeattrfunc(node):
    str = f'label="{node.name}\nV: {node.value}\nN: {node.visits}"'
    if node.to_play == 1:
        str += ", shape=box"
    return str


def edgeattrfunc(parent, child):
    return f'label="A: {child.last_action} \np: {str(child.prior)}\nR: {child.reward}"'


_partial_dot_exporter = functools.partial(
    anytree.exporter.UniqueDotExporter,
    nodenamefunc=lambda node: node.id,
    nodeattrfunc=nodeattrfunc,
    edgeattrfunc=edgeattrfunc,
)


def anytree_to_png(anytree_root, file_path):
    _partial_dot_exporter(anytree_root).to_picture(file_path)


def anytree_display_in_notebook(anytree_root):
    from IPython import display

    anytree_to_png(anytree_root, "/tmp/anytree_node.png")
    image = display.Image(filename="/tmp/anytree_node.png")
    return display.display(image)


def anytree_to_json(anytree_root, file_path):
    json_s = anytree.exporter.JsonExporter(indent=2, sort_keys=True).export(
        anytree_root
    )
    with open(file_path, "w") as f:
        f.write(json_s)


def anytree_to_numpy(anytree_root):
    import PIL

    anytree_to_png(anytree_root, "/tmp/anytree_to_numpy.png")
    arr = np.asarray(PIL.Image.open("/tmp/anytree_to_numpy.png"))
    return arr


def anytree_to_text(anytree_root) -> str:
    s = ""
    for row in anytree.RenderTree(anytree_root):
        s += (
            f"{row.pre} A:{row.node.last_action} R:{row.node.reward} V:{row.node.value}"
            f"[{row.node.name} ({row.node.to_play})] N:{row.node.visits}"
        )
        s += "\n"
    return s


def anytree_filter_node(anytree_node, filter_fn):
    anytree_node.children = [c for c in anytree_node.children if filter_fn(c)]
    for c in anytree_node.children:
        anytree_filter_node(c, filter_fn)
