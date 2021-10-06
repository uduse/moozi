from dataclasses import InitVar, dataclass, field
from typing import Any, Awaitable, Callable, Dict, NamedTuple, Optional

import inspect
import jax.numpy as jnp
import numpy as np
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
        nn_output = await self.recurr_inf_fn(node.parent.hidden_state, action)
        node.expand_node(nn_output, self.all_actions_mask)
        node.backpropagate(float(nn_output.value), self.discount)

    def __del__(self):
        """TODO: relase recurrent states stored remotely."""
