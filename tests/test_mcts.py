import asyncio
import functools
import pytest
from dataclasses import dataclass
from typing import Callable

import jax
import numpy as np
import ray
import tree
from acme.jax.networks.base import NetworkOutput
from acme.utils.tree_utils import unstack_sequence_fields
from moozi.batching_layer import BatchingClient, BatchingLayer
from moozi.core import PolicyFeed
from moozi.nn import NNOutput, NNModel
from moozi.policy.mcts import MCTSAsync
from moozi.utils import as_coroutine


# def test_node():
#     node = Node(prior=0.5, player=0)
#     nn_output = NNOutput(
#         value=0.5, reward=1, policy_logits=np.array([0.1, 0.1, 0.8]), hidden_state=None
#     )
#     node.expand_node(
#         hidden_state=nn_output.hidden_state,
#         reward=nn_output.reward,
#         policy_logits=nn_output.policy_logits,
#         legal_actions_mask=np.array([1, 1, 1]),
#         next_player=get_next_player(SearchStrategy.TWO_PLAYER, node.player),
#     )
#     assert node


async def test_async_mcts_with_regular_inf_fns(
    env_spec, model: NNModel, params, state, policy_feed: PolicyFeed
):
    is_training = False

    def root_inf_fn(feats):
        out, _ = model.root_inference_unbatched(params, state, feats, is_training)
        return out

    def trans_inf_fn(feats):
        out, _ = model.trans_inference_unbatched(params, state, feats, is_training)
        return out

    mcts_async = MCTSAsync(
        root_inf_fn=as_coroutine(root_inf_fn),
        trans_inf_fn=as_coroutine(trans_inf_fn),
        dim_action=env_spec.actions.num_values,
    )

    root = await mcts_async.run(policy_feed)
    assert root


async def test_async_mcts_with_async_inf_fns(
    env_spec, model: NNModel, params, state, policy_feed: PolicyFeed
):
    is_training = False

    async def root_inf_fn(feats):
        out, _ = model.root_inference_unbatched(params, state, feats, is_training)
        return out

    async def trans_inf_fn(feats):
        out, _ = model.trans_inference_unbatched(params, state, feats, is_training)
        return out

    mcts_async = MCTSAsync(
        root_inf_fn=root_inf_fn,
        trans_inf_fn=trans_inf_fn,
        num_simulations=10,
        dim_action=env_spec.actions.num_values,
    )

    root = await mcts_async.run(policy_feed)
    assert root


async def test_async_mcts_with_ray(
    model: NNModel, params, state, policy_feed: PolicyFeed, env_spec
):
    ray.init(ignore_reinit_error=True)

    @dataclass
    class SimpleInferenceServer:
        _root_inf = functools.partial(
            jax.jit(model.root_inference_unbatched, static_argnums=3),
            params,
            state,
            is_training=True,
        )
        _trans_inf = functools.partial(
            jax.jit(model.trans_inference_unbatched, static_argnums=3),
            params,
            state,
            is_training=True,
        )

        def root_inf(self, feats) -> NNOutput:
            return self._root_inf(feats)[0]

        def trans_inf(self, feats) -> NNOutput:
            return self._trans_inf(feats)[0]

    inf_server = ray.remote(SimpleInferenceServer).remote()

    mcts_async = MCTSAsync(
        root_inf_fn=inf_server.root_inf.remote,
        trans_inf_fn=inf_server.trans_inf.remote,
        dim_action=env_spec.actions.num_values,
    )

    root = await mcts_async.run(policy_feed)
    assert root


# # TODO: fix this test
# @with_trio_asyncio
# async def test_async_mcts_with_ray_and_batching(
#     model: NNModel, params, state, policy_feed: PolicyFeed, env_spec
# ):
#     @ray.remote
#     @dataclass
#     class SimpleInferenceServer:
#         _root_inf: Callable = functools.partial(
#             jax.jit(model.root_inference), params, state, is_training
#         )
#         _trans_inf: Callable = functools.partial(
#             jax.jit(model.trans_inference), params, state, is_training
#         )

#         def root_inf(self, feats):
#             batch_size = len(feats)
#             results = self._root_inf(np.array(feats))
#             results = tree.map_structure(np.array, results)
#             return unstack_sequence_fields(results, batch_size)

#         def recurr_inf(self, inputs):
#             batch_size = len(inputs)
#             hidden_states = np.array(list(map(operator.itemgetter(0), inputs)))
#             actions = np.array(list(map(operator.itemgetter(1), inputs)))
#             results = self._trans_inf(hidden_states, actions)
#             results = tree.map_structure(np.array, results)
#             return unstack_sequence_fields(results, batch_size)

#     server = SimpleInferenceServer.remote()

#     async def mcts_init_inf_fn(x):
#         return await aio_as_trio(server.init_inf.remote(x))

#     async def mcts_recurr_inf_fn(x):
#         return await aio_as_trio(server.recurr_inf.remote(x))

#     bl_init_inf = BatchingLayer(max_batch_size=5, process_fn=mcts_init_inf_fn)
#     bl_recurr_inf = BatchingLayer(max_batch_size=5, process_fn=mcts_recurr_inf_fn)

#     num_mcts = 10
#     mctses = []
#     for _ in range(num_mcts):
#         mcts = MCTSAsync(
#             root_inf_fn=bl_init_inf.spawn_client().request,
#             trans_inf_fn=bl_recurr_inf.spawn_client().request,
#             num_simulations=10,
#             dim_action=env_spec.actions.num_values,
#         )
#         mctses.append(mcts)

#     async with trio.open_nursery() as n:
#         n.start_soon(bl_init_inf.start_processing)
#         n.start_soon(bl_recurr_inf.start_processing)

#         async with bl_init_inf.open_context(), bl_recurr_inf.open_context():
#             async with trio.open_nursery() as nn:
#                 for mcts in mctses:
#                     nn.start_soon(mcts.run, policy_feed)

#     assert mctses
