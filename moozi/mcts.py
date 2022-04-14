import copy
import functools
import uuid
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional, Tuple

import anytree
import jax
import jax.numpy as jnp
import numpy as np
from loguru import logger
from moozi import BASE_PLAYER, PolicyFeed, link, MinMaxStats
from moozi.nn import NNModel, NNOutput, NNSpec, RootFeatures, TransitionFeatures


def get_next_player(num_players: int, to_play: int) -> int:
    if num_players == 2:
        if to_play == 0:
            return 1
        elif to_play == 1:
            return 0
        else:
            raise ValueError(f"Invalid to_play: {to_play}")
    elif num_players == 1:
        return 0

    else:
        raise NotImplementedError


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

    # TODO: move all methods to mcts class
    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        else:
            return self.value_sum / self.visit_count

    @property
    def is_expanded(self) -> bool:
        return len(self.children) > 0


# @dataclass
# class SearchState:
#     root: Node
#     min_max_stats: MinMaxStats


@dataclass
class MCTSAsync:
    root_inf_fn: Callable[[RootFeatures], Awaitable[NNOutput]]
    trans_inf_fn: Callable[[TransitionFeatures], Awaitable[NNOutput]]
    dim_action: int

    num_players: int = 1
    num_simulations: int = 10
    allow_all_actions_mask: np.ndarray = field(init=False)
    discount: float = 1.0

    dirichlet_alpha: float = 0.2
    frac: float = 0.2

    known_bound_min: Optional[float] = None
    known_bound_max: Optional[float] = None

    def __post_init__(self):
        self.allow_all_actions_mask = np.ones((self.dim_action,), dtype=np.int32)

    async def run(self, feed: PolicyFeed) -> Node:
        root, min_max_stats = await self.init_search_state(feed)
        self.add_exploration_noise(root, self.dirichlet_alpha, self.frac)

        for _ in range(self.num_simulations):
            await self.simulate_once(root, min_max_stats)

        return root

    async def init_search_state(self, feed: PolicyFeed) -> Tuple[Node, MinMaxStats]:
        if (self.known_bound_min is not None) and (self.known_bound_max is not None):
            min_max_stats = MinMaxStats(self.known_bound_min, self.known_bound_max)
        else:
            min_max_stats = MinMaxStats()
        root_feats = RootFeatures(
            obs=feed.stacked_frames, player=np.array(feed.to_play)
        )
        root_nn_output = await self.root_inf_fn(root_feats)
        root = Node(0, player=feed.to_play, name="s")
        self.expand_node(
            root,
            hidden_state=root_nn_output.hidden_state,
            reward=0.0,
            policy_logits=root_nn_output.policy_logits,
            legal_actions_mask=feed.legal_actions_mask,
            next_player=get_next_player(self.num_players, feed.to_play),
        )
        self.backpropagate(
            node=root, value=float(root_nn_output.value), min_max_stats=min_max_stats
        )
        return (root, min_max_stats)

    async def simulate_once(self, root: Node, min_max_stats: MinMaxStats):
        action, leaf = self.select_leaf(root, min_max_stats)
        assert leaf.parent

        trans_feats = TransitionFeatures(
            hidden_state=leaf.parent.hidden_state, action=np.array(action)
        )
        leaf_nn_output = await self.trans_inf_fn(trans_feats)

        reward = float(leaf_nn_output.reward)
        value = float(leaf_nn_output.value)

        self.expand_node(
            leaf,
            hidden_state=leaf_nn_output.hidden_state,
            reward=reward,
            policy_logits=leaf_nn_output.policy_logits,
            legal_actions_mask=self.allow_all_actions_mask,
            next_player=get_next_player(self.num_players, leaf.player),
        )

        self.backpropagate(leaf, value, min_max_stats)

    def expand_node(
        self,
        node,
        hidden_state: np.ndarray,
        reward: float,
        policy_logits: np.ndarray,
        legal_actions_mask: np.ndarray,
        next_player: int,
    ):

        node.hidden_state = hidden_state
        node.last_reward = reward
        action_probs = self._compute_action_probs(policy_logits, legal_actions_mask)

        for action, prob in enumerate(action_probs):
            if prob > 0.0:
                node.children[action] = Node(
                    prior=prob,
                    player=next_player,
                    parent=node,
                    name=node.name + f"_{action}",
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

    def ucb_score(
        self,
        parent: "Node",
        child: "Node",
        min_max_stats: MinMaxStats,
    ) -> float:
        # TODO: obviously these constants here should be a parameter
        pb_c_base = 19652.0
        pb_c_init = 1.25

        pb_c = np.log((parent.visit_count + pb_c_base + 1) / pb_c_base) + pb_c_init
        pb_c *= np.sqrt(parent.visit_count) / (child.visit_count + 1)
        prior_score = pb_c * child.prior

        if child.visit_count > 0:
            value_score = child.last_reward + self.discount * min_max_stats.normalize(
                child.value
            )
            # max(-scores) if the parent player is not the BASE_PLAYER
            if parent.player != BASE_PLAYER:
                value_score = -value_score
        else:
            value_score = 0.0

        return prior_score + value_score

    def select_leaf(self, root: Node, min_max_stats: MinMaxStats) -> Tuple[int, Node]:
        node = root
        while node.is_expanded:
            action, node = self.select_child(node, min_max_stats)
        return action, node

    def select_child(self, node: Node, min_max_stats: MinMaxStats):
        scores = []
        for action, child in node.children.items():
            ucb_score = self.ucb_score(
                parent=node, child=child, min_max_stats=min_max_stats
            )
            scores.append((ucb_score, action, child))

        # TODO: break ties randomly
        _, action, child = max(scores)
        return action, child

    def add_exploration_noise(
        self, node, dirichlet_alpha: float = 0.2, frac: float = 0.2
    ):
        actions = list(node.children.keys())
        noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
        for a, n in zip(actions, noise):
            node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac

    def backpropagate(self, node: Node, value: float, min_max_stats: MinMaxStats):
        while True:
            node.value_sum += value
            node.visit_count += 1
            min_max_stats.update(node.value)

            if node.parent:
                value = node.last_reward + value * self.discount
                node = node.parent
            else:
                break

    def get_children_visit_counts(self, node: Node):
        visit_counts = {}
        for action, child in node.children.items():
            visit_counts[action] = child.visit_count
        return visit_counts

    def get_children_values(self, node):
        values = {}
        for action, child in node.children.items():
            values[action] = child.value
        return values

    def get_children_visit_counts_as_probs(self, node: Node):
        action_probs = np.zeros((self.dim_action,), dtype=np.float32)
        for a, visit_count in self.get_children_visit_counts(node).items():
            action_probs[a] = visit_count
        action_probs /= np.sum(action_probs)
        return action_probs


@link
@dataclass
class Planner:
    num_simulations: int
    known_bound_min: Optional[float]
    known_bound_max: Optional[float]
    include_tree: bool = False

    # def __post__init__(self):
    #     logger.remove(0)
    #     logger.add("logs/planner.log", level="DEBUG")

    async def __call__(
        self,
        is_last,
        legal_actions_mask,
        policy_feed,
        root_inf_fn,
        trans_inf_fn,
    ):
        if not is_last:
            mcts = MCTSAsync(
                root_inf_fn=root_inf_fn,
                trans_inf_fn=trans_inf_fn,
                dim_action=legal_actions_mask.size,
                num_simulations=self.num_simulations,
                known_bound_min=self.known_bound_min,
                known_bound_max=self.known_bound_max,
            )
            mcts_root = await mcts.run(policy_feed)
            action_probs = mcts.get_children_visit_counts_as_probs(
                mcts_root,
            )

            # if np.isnan(action_probs).any():
            #     breakpoint()
            # logger.debug(f"{action_probs=}")
            if self.include_tree:
                return dict(
                    action_probs=action_probs,
                    mcts_root=copy.deepcopy(mcts_root),
                )
            else:
                return dict(action_probs=action_probs)


def sample_action(action_probs, temperature=1.0):
    try:
        log_probs = np.log(np.clip(action_probs, 1e-10, None)) / temperature
        with_temperature = np.exp(log_probs) / np.sum(np.exp(log_probs))
        return np.random.choice(np.arange(len(with_temperature)), p=with_temperature)
    except Exception:
        logger.error(f"{action_probs=}, {log_probs=}, {with_temperature=}")


@link
@dataclass
class ActionSamplerLaw:
    temperature: float = 1.0

    def __call__(self, action_probs):
        action = sample_action(action_probs, temperature=self.temperature)
        return dict(action=action)


def temp(arr, temperature=1.0):
    log = np.log(arr) / temperature
    return np.exp(log) / np.sum(np.exp(log))


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
