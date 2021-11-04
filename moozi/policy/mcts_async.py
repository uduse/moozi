from dataclasses import InitVar, dataclass
from typing import Awaitable, Callable

import inspect
from jax._src.numpy.lax_numpy import isin
import jax.numpy as jnp
import numpy as np
from moozi import link
from moozi.batching_layer import BatchingClient
from moozi.nn import NeuralNetwork, NeuralNetworkSpec, NNOutput, get_network
from moozi.policy.mcts_core import Node, SearchStrategy, next_player
from moozi import PolicyFeed
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

        for _ in range(self.num_simulations):
            await self.simulate_once(root)

        return root

    async def get_root(self, feed: PolicyFeed) -> Node:
        root_nn_output = await self.init_inf_fn(feed.stacked_frames)
        root = Node(0)
        to_play = next_player(self.strategy, feed.to_play)
        root.expand_node(root_nn_output, feed.legal_actions_mask, to_play)
        root.add_exploration_noise()
        return root

    async def simulate_once(self, root: Node):
        action, node = root.select_leaf()
        assert node.parent
        nn_output = await self.recurr_inf_fn((node.parent.hidden_state, action))
        to_play = next_player(self.strategy, node.to_play)
        node.expand_node(nn_output, self.all_actions_mask, to_play)
        node.backpropagate(float(nn_output.value), self.discount, to_play)


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
