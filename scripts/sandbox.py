# %%
import functools
from dataclasses import dataclass
from typing import NamedTuple

import chex
import graphviz
import haiku as hk
import jax
import jax.numpy as jnp
import tree
from acme.jax.utils import add_batch_dim, squeeze_batch_dim
from moozi.nn import (
    NeuralNetwork,
    NNOutput,
    NNSpec,
    RootInferenceFeatures,
    TransitionInferenceFeatures,
)

# %%




# # %%
# num_stacked_frames = 4
# frame_shape = (5, 5, 3)
# dim_repr = 16
# dim_action = 7
# stacked_frames_shape = (num_stacked_frames,) + frame_shape
# spec = NeuralNetworkSpec(
#     stacked_frames_shape=stacked_frames_shape,
#     dim_repr=dim_repr,
#     dim_action=dim_action,
#     extra={
#         "repr_net_num_blocks": 1,
#         "pred_trunk_num_blocks": 1,
#         "dyna_trunk_num_blocks": 1,
#         "dyna_hidden_num_blocks": 1,
#     },
# )
# nn = build_network(spec)
# rng = jax.random.PRNGKey(0)

# # %%
# params, state = nn.init_network(rng)

# # %%
# root_inf_feats = RootInferenceFeatures(
#     stacked_frames=jnp.ones(stacked_frames_shape), player=jnp.array(0)
# )
# nn_out, new_state = nn.root_inference_unbatched(params, state, root_inf_feats)

# # %%
# trans_inf_feats = TransitionInferenceFeatures(
#     hidden_state=nn_out.hidden_state, action=jnp.ones((1,))
# )
# nn_out, new_state = nn.trans_inference_unbatched(params, state, trans_inf_feats)

# # %%
# html_info = hk.experimental.tabulate(
#     nn.root_inference_unbatched,
#     filters=["has_params", "has_output"],
#     columns=(
#         "module",
#         "config",
#         "owned_params",
#         "input",
#         "output",
#         "params_size",
#         "params_bytes",
#     ),
#     tabulate_kwargs={"tablefmt": "html"},
# )(params, state, root_inf_feats)
# html_info._repr_html_()
