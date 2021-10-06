import functools
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, NamedTuple, Optional

import anytree
import jax
import jax.numpy as jnp
import numpy as np
import ray.util.queue
import rlax
import trio
from moozi.nn import NeuralNetwork, NNOutput, NeuralNetworkSpec, get_network
from moozi.policies.policy import PolicyFeed, PolicyFn, PolicyResult

_safe_epsilon_softmax = jax.jit(rlax.safe_epsilon_softmax(1e-7, 1).probs, backend="cpu")




class MonteCarloTreeSearch:
    def __init__(
        self,
        requester,
        num_simulations: int,
        discount: float,
        dim_action: int,
    ) -> None:
        super().__init__()
        self._requester = requester
        self._num_simulations = num_simulations
        self._discount = discount
        self._all_actions_mask = np.ones((dim_action,), dtype=np.int32)

    async def __call__(self, feed: PolicyFeed) -> Node:
        root = await self.get_root(feed)

        for _ in range(self._num_simulations):
            await self.simulate_once(root)

        return root

    async def get_root(self, feed: PolicyFeed) -> Node:
        root_nn_output = await self._requester.initial_inference(feed.stacked_frames)
        root = Node(0)
        root.expand_node(root_nn_output, feed.legal_actions_mask)
        root.add_exploration_noise()
        return root

    async def simulate_once(
        self,
        root: Node,
    ):
        action, node = root.select_leaf()
        assert node.parent
        nn_output = await self._requester.recurrent_inference(
            node.parent.hidden_state_id, action
        )
        node.expand_node(nn_output, self._all_actions_mask)
        node.backpropagate(float(nn_output.value), self._discount)

    @property
    def all_actions_mask(self):
        return self._all_actions_mask

    def __del__(self):
        """TODO: relase recurrent states stored remotely."""


# def get_uuid():
#     return uuid.uuid4().hex[:8]


# def convert_to_anytree(node: Node, anytree_node=None, action="_"):
#     if node.visit_count > 0:
#         anytree_node = anytree.Node(
#             id=get_uuid(),
#             name=action,
#             parent=anytree_node,
#             prior=str(np.round(node.prior, 3)),
#             reward=str(np.round(node.reward, 3)),
#             value=str(np.round(node.value, 3)),
#             visits=str(np.round(node.visit_count, 3)),
#         )
#         for action, child in node.children.items():
#             convert_to_anytree(child, anytree_node, action)
#         return anytree_node
#     else:
#         return anytree_node


# def nodeattrfunc(node):
#     return f'label="V: {node.value}\nN: {node.visits}"'


# def edgeattrfunc(parent, child):
#     return f'label="A: {child.name} \np: {child.prior}\nR: {child.reward}"'


# _partial_dot_exporter = functools.partial(
#     anytree.exporter.UniqueDotExporter,
#     nodenamefunc=lambda node: node.id,
#     nodeattrfunc=nodeattrfunc,
#     edgeattrfunc=edgeattrfunc,
# )


# def anytree_to_png(anytree_root, file_path):
#     _partial_dot_exporter(anytree_root).to_picture(file_path)


# def anytree_to_json(anytree_root, file_path):
#     json_s = anytree.exporter.JsonExporter(indent=2, sort_keys=True).export(
#         anytree_root
#     )
#     with open(file_path, "w") as f:
#         f.write(json_s)


# def train_policy_fn(mcts: MonteCarloTreeSearch, feed: PolicyFeed) -> PolicyResult:
#     mcts_tree = mcts(feed)

#     action_probs = np.zeros_like(mcts.all_actions_mask, dtype=np.float32)
#     for a, visit_count in mcts_tree.get_children_visit_counts().items():
#         action_probs[a] = visit_count
#     action_probs /= np.sum(action_probs)

#     action, _ = mcts_tree.select_child()
#     return PolicyResult(
#         action=np.array(action, dtype=np.int32),
#         extras={"tree": mcts_tree, "action_probs": action_probs},
#     )


# def eval_policy_fn(mcts: MonteCarloTreeSearch, feed: PolicyFeed) -> PolicyResult:
#     policy_result = train_policy_fn(mcts, feed)

#     mcts_tree = policy_result.extras["tree"]

#     child_values = np.zeros_like(mcts.all_actions_mask, dtype=np.float32)
#     for action, value in mcts_tree.get_children_values().items():
#         child_values[action] = value

#     policy_result.extras.update(
#         {
#             "child_values": child_values,
#         }
#     )
#     return policy_result
