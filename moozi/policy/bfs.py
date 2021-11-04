# from typing import NamedTuple
# from acme.core_test import StepCountingLearner
# import chex
# import jax
# import jax.numpy as jnp
# import numpy as np
# import rlax
# from acme.jax.utils import add_batch_dim, squeeze_batch_dim
# from acme.jax.variable_utils import VariableClient
# from jax.ops import index_add, index
# from moozi.nn import NeuralNetwork, NNOutput

# from .policy import PolicyFn, PolicyFeed, PolicyResult


# def make_bfs_fn(network, num_unroll_steps):
#     @jax.jit
#     def bfs_fn(params, feed: PolicyFeed) -> Node:
#         action_space_size = jnp.size(feed.legal_actions_mask)

#         root_network_output = network.initial_inference_unbatched(
#             params, feed.stacked_frames
#         )

#         root_node = Node(
#             network_output=root_network_output,
#             prior=jnp.array(0),
#             children=[],
#         )
#         frontier = [root_node]

#         for _ in range(num_unroll_steps):
#             next_frontier = []
#             for node in frontier:
#                 children = _expand_node(network, params, node, action_space_size)
#                 next_frontier.extend(children)
#             frontier = next_frontier
#         return root_node

#     return bfs_fn