from dataclasses import InitVar, dataclass
from typing import Awaitable, Callable

import inspect
import jax.numpy as jnp
import numpy as np
from moozi import link
from moozi.batching_layer import BatchingClient
from moozi.nn import NeuralNetwork, NeuralNetworkSpec, NNOutput, get_network
from moozi.policies.mcts_core import Node
from moozi.policies.policy import PolicyFeed, PolicyFn, PolicyResult
from moozi.utils import as_coroutine


@dataclass
class MCTSAsync:
    init_inf_fn: Callable[..., Awaitable[NNOutput]]
    recurr_inf_fn: Callable[..., Awaitable[NNOutput]]
    num_simulations: int
    dim_action: InitVar[int]
    all_actions_mask: np.ndarray = None
    discount: float = 1.0

    def __post_init__(self, dim_action):
        self.all_actions_mask = np.ones((dim_action,), dtype=np.int32)
        if not inspect.iscoroutinefunction(self.init_inf_fn):
            self.init_inf_fn = as_coroutine(self.init_inf_fn)
        if not inspect.iscoroutinefunction(self.recurr_inf_fn):
            self.recurr_inf_fn = as_coroutine(self.recurr_inf_fn)

    async def __call__(self, feed: PolicyFeed) -> Node:
        root = await self.get_root(feed)

        for _ in range(self.num_simulations):
            await self.simulate_once(root)

        return root

    async def get_root(self, feed: PolicyFeed) -> Node:
        root_nn_output = await self.init_inf_fn(feed.stacked_frames)
        root = Node(0)
        root.expand_node(root_nn_output, feed.legal_actions_mask)
        root.add_exploration_noise()
        return root

    async def simulate_once(self, root: Node):
        action, node = root.select_leaf()
        assert node.parent
        nn_output = await self.recurr_inf_fn((node.parent.hidden_state, action))
        node.expand_node(nn_output, self.all_actions_mask)
        node.backpropagate(float(nn_output.value), self.discount)

    def __del__(self):
        """TODO: relase recurrent states stored remotely."""


def make_async_planner_law(init_inf_fn, recurr_inf_fn, dim_actions, num_simulations=10):
    mcts = MCTSAsync(
        init_inf_fn=init_inf_fn,
        recurr_inf_fn=recurr_inf_fn,
        num_simulations=num_simulations,
        dim_action=dim_actions,
    )

    @link
    async def planner(is_last, stacked_frames, legal_actions_mask):
        if not is_last:
            feed = PolicyFeed(
                stacked_frames=stacked_frames,
                legal_actions_mask=legal_actions_mask,
                random_key=None,
            )
            mcts_tree = await mcts(feed)
            action, _ = mcts_tree.select_child()

            action_probs = np.zeros((3,), dtype=np.float32)
            for a, visit_count in mcts_tree.get_children_visit_counts().items():
                action_probs[a] = visit_count
            action_probs /= np.sum(action_probs)
            return dict(action=action, action_probs=action_probs)

    return planner
