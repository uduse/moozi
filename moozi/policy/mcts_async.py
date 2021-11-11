from dataclasses import InitVar, dataclass
from typing import Awaitable, Callable

import inspect
from jax._src.numpy.lax_numpy import isin
import jax.numpy as jnp
import numpy as np
from moozi import link
from moozi.batching_layer import BatchingClient
from moozi.nn import NeuralNetwork, NeuralNetworkSpec, NNOutput, get_network
from moozi.policy.mcts_core import (
    Node,
    SearchStrategy,
    get_next_player,
    get_prev_player,
)
from moozi import PolicyFeed, BASE_PLAYER
from moozi.utils import as_coroutine


@dataclass
class MCTSAsync:
    dim_action: InitVar[int]

    init_inf_fn: Callable[..., Awaitable[NNOutput]]
    recurr_inf_fn: Callable[..., Awaitable[NNOutput]]
    num_simulations: int
    all_actions_mask: np.ndarray = None
    discount: float = 1.0

    strategy: SearchStrategy = SearchStrategy.TWO_PLAYER

    def __post_init__(self, dim_action):
        self.all_actions_mask = np.ones((dim_action,), dtype=np.int32)
        if not inspect.iscoroutinefunction(self.init_inf_fn):
            self.init_inf_fn = as_coroutine(self.init_inf_fn)
        if not inspect.iscoroutinefunction(self.recurr_inf_fn):
            self.recurr_inf_fn = as_coroutine(self.recurr_inf_fn)

    async def run(self, feed: PolicyFeed) -> Node:
        root = await self.get_root(feed)
        root.add_exploration_noise()

        for _ in range(self.num_simulations):
            await self.simulate_once(root)

        return root

    async def get_root(self, feed: PolicyFeed) -> Node:
        root_nn_output = await self.init_inf_fn(feed.stacked_frames)
        root = Node(0)
        next_player = get_next_player(self.strategy, feed.to_play)
        root.expand_node(
            hidden_state=root_nn_output.hidden_state,
            reward=0.0,
            policy_logits=root_nn_output.policy_logits,
            legal_actions_mask=feed.legal_actions_mask,
            next_player=next_player,
        )

        value = self._reorient(float(root_nn_output.value), root.player, feed.to_play)
        root.backpropagate(
            value=value,
            discount=self.discount,
            root_player=root.player,
        )
        return root

    async def simulate_once(self, root: Node):
        action, leaf = root.select_leaf()
        assert leaf.parent
        leaf_nn_output = await self.recurr_inf_fn((leaf.parent.hidden_state, action))

        reward = self._reorient(
            float(leaf_nn_output.reward),
            root_player=root.player,
            target_player=get_prev_player(self.strategy, leaf.player),
        )

        value = self._reorient(
            float(leaf_nn_output.value),
            root_player=root.player,
            target_player=root.player,
        )

        leaf.expand_node(
            hidden_state=leaf_nn_output.hidden_state,
            reward=reward,
            policy_logits=leaf_nn_output.policy_logits,
            legal_actions_mask=self.all_actions_mask,
            next_player=get_next_player(self.strategy, leaf.player),
        )

        leaf.backpropagate(
            value=value,
            discount=self.discount,
            root_player=root.player,
        )

    def _reorient(self, item: float, root_player: int, target_player: int) -> float:
        """Reorient value or reward to the root_player's perspective."""
        is_same_player = root_player == target_player
        if (root_player == BASE_PLAYER and not is_same_player) or (
            root_player != BASE_PLAYER and is_same_player
        ):
            return -item
        else:
            return item


def make_async_planner_law(init_inf_fn, recurr_inf_fn, dim_actions, num_simulations=10):
    mcts = MCTSAsync(
        init_inf_fn=init_inf_fn,
        recurr_inf_fn=recurr_inf_fn,
        num_simulations=num_simulations,
        dim_action=dim_actions,
    )

    @link
    async def planner(is_last, policy_feed):
        if not is_last:
            mcts_tree = await mcts.run(policy_feed)
            action, _ = mcts_tree.select_child()

            action_probs = np.zeros((dim_actions,), dtype=np.float32)
            for a, visit_count in mcts_tree.get_children_visit_counts().items():
                action_probs[a] = visit_count
            action_probs /= np.sum(action_probs)

            return dict(action=action, action_probs=action_probs)

    return planner
