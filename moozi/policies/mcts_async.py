from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, NamedTuple, Optional

import jax.numpy as jnp
import numpy as np
from moozi.batching_layer import BatchingClient
from moozi.nn import NeuralNetwork, NeuralNetworkSpec, NNOutput, get_network
from moozi.policies.mcts_core import Node
from moozi.policies.policy import PolicyFeed, PolicyFn, PolicyResult


class MCTSAsync:
    def __init__(
        self,
        ini_inf_fn: Callable[..., Awaitable[NNOutput]],
        recurr_inf_fn: Callable[..., Awaitable[NNOutput]],
        num_simulations: int,
        discount: float,
        dim_action: int,
    ) -> None:
        self._init_inf = ini_inf_fn
        self._recurr_inf = recurr_inf_fn
        self._num_simulations = num_simulations
        self._discount = discount
        self._all_actions_mask = np.ones((dim_action,), dtype=np.int32)

    async def __call__(self, feed: PolicyFeed) -> Node:
        root = await self.get_root(feed)

        for _ in range(self._num_simulations):
            await self.simulate_once(root)

        return root

    async def get_root(self, feed: PolicyFeed) -> Node:
        root_nn_output = await self._init_inf(feed.stacked_frames)
        root = Node(0)
        root.expand_node(root_nn_output, feed.legal_actions_mask)
        root.add_exploration_noise()
        return root

    async def simulate_once(self, root: Node):
        action, node = root.select_leaf()
        assert node.parent
        nn_output = await self._recurr_inf((node.parent.hidden_state, action))
        node.expand_node(nn_output, self._all_actions_mask)
        node.backpropagate(float(nn_output.value), self._discount)

    @property
    def all_actions_mask(self):
        return self._all_actions_mask

    def __del__(self):
        """TODO: relase recurrent states stored remotely."""
