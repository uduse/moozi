import functools
import pytest
import operator
from dataclasses import dataclass
from typing import Callable

import jax
import numpy as np
import ray
import tree
import trio
import trio_asyncio
from acme.jax.networks.base import NetworkOutput
from acme.utils.tree_utils import unstack_sequence_fields
from moozi.batching_layer import BatchingClient, BatchingLayer
from moozi.core import PolicyFeed
from moozi.nn import NNOutput, NeuralNetwork
from moozi.policy.mcts_async import MCTSAsync
from moozi.policy.mcts_core import (
    Node,
    anytree_to_text,
    convert_to_anytree,
    get_next_player,
    SearchStrategy,
    reorient,
)
from moozi.utils import as_coroutine
from trio_asyncio import aio_as_trio

from tests.utils import with_trio_asyncio


def test_node():
    node = Node(prior=0.5, player=0)
    nn_output = NNOutput(
        value=0.5, reward=1, policy_logits=np.array([0.1, 0.1, 0.8]), hidden_state=None
    )
    node.expand_node(
        hidden_state=nn_output.hidden_state,
        reward=nn_output.reward,
        policy_logits=nn_output.policy_logits,
        legal_actions_mask=np.array([1, 1, 1]),
        next_player=get_next_player(SearchStrategy.TWO_PLAYER, node.player),
    )
    assert node


async def test_async_mcts(
    network: NeuralNetwork, params, policy_feed: PolicyFeed, env_spec
):
    async def init_inf(frames):
        return network.initial_inference_unbatched(params, frames)

    def recurr_inf(inputs):
        hidden_state, action = inputs
        return network.recurrent_inference_unbatched(params, hidden_state, action)

    mcts_async = MCTSAsync(
        init_inf_fn=init_inf,
        recurr_inf_fn=recurr_inf,
        num_simulations=10,
        dim_action=env_spec.actions.num_values,
    )

    root = await mcts_async.run(policy_feed)
    assert root


@with_trio_asyncio
async def test_async_mcts_with_ray(
    network: NeuralNetwork, params, policy_feed: PolicyFeed, env_spec, init_ray
):
    @dataclass
    class SimpleInferenceServer:
        _init_inf_fn: Callable = functools.partial(
            jax.jit(network.initial_inference_unbatched), params
        )
        _recurr_inf_fn: Callable = functools.partial(
            jax.jit(network.recurrent_inference_unbatched), params
        )

        def init_inf(self, frames):
            return self._init_inf_fn(frames)

        def recurr_inf(self, hidden_states, action):
            return self._recurr_inf_fn(hidden_states, action)

    inf_server = ray.remote(SimpleInferenceServer).remote()

    async def mcts_init_inf_fn(x):
        return await aio_as_trio(inf_server.init_inf.remote(x))

    async def mcts_recurr_inf_fn(inputs):
        x, y = inputs
        return await aio_as_trio(inf_server.recurr_inf.remote(x, y))

    mcts_async = MCTSAsync(
        init_inf_fn=mcts_init_inf_fn,
        recurr_inf_fn=mcts_recurr_inf_fn,
        num_simulations=10,
        dim_action=env_spec.actions.num_values,
    )

    root = await mcts_async.run(policy_feed)
    assert root


@with_trio_asyncio
async def test_async_mcts_with_ray_and_batching(
    network: NeuralNetwork, params, policy_feed: PolicyFeed, env_spec, init_ray
):
    @dataclass
    class SimpleInferenceServer:
        _init_inf_fn: Callable = functools.partial(
            jax.jit(network.initial_inference), params
        )
        _recurr_inf_fn: Callable = functools.partial(
            jax.jit(network.recurrent_inference), params
        )

        def init_inf(self, frames):
            batch_size = len(frames)
            results = self._init_inf_fn(np.array(frames))
            results = tree.map_structure(np.array, results)
            return unstack_sequence_fields(results, batch_size)

        def recurr_inf(self, inputs):
            batch_size = len(inputs)
            hidden_states = np.array(list(map(operator.itemgetter(0), inputs)))
            actions = np.array(list(map(operator.itemgetter(1), inputs)))
            results = self._recurr_inf_fn(hidden_states, actions)
            results = tree.map_structure(np.array, results)
            return unstack_sequence_fields(results, batch_size)

    inf_server = ray.remote(SimpleInferenceServer).remote()

    async def mcts_init_inf_fn(x):
        return await aio_as_trio(inf_server.init_inf.remote(x))

    async def mcts_recurr_inf_fn(x):
        return await aio_as_trio(inf_server.recurr_inf.remote(x))

    bl_init_inf = BatchingLayer(max_batch_size=5, process_fn=mcts_init_inf_fn)
    bl_recurr_inf = BatchingLayer(max_batch_size=5, process_fn=mcts_recurr_inf_fn)

    num_mcts = 10
    mctses = []
    for _ in range(num_mcts):
        mcts = MCTSAsync(
            init_inf_fn=bl_init_inf.spawn_client().request,
            recurr_inf_fn=bl_recurr_inf.spawn_client().request,
            num_simulations=10,
            dim_action=env_spec.actions.num_values,
        )
        mctses.append(mcts)

    async with trio.open_nursery() as n:
        n.start_soon(bl_init_inf.start_processing)
        n.start_soon(bl_recurr_inf.start_processing)

        async with bl_init_inf.open_context(), bl_recurr_inf.open_context():
            async with trio.open_nursery() as nn:
                for mcts in mctses:
                    nn.start_soon(mcts.run, policy_feed)

    assert mctses


x = 10


@pytest.mark.parametrize(
    "item,root_player,target_player,expected",
    [
        (x, 0, 0, x),
        (x, 0, 1, -x),
        (x, 1, 0, -x),
        (x, 1, 1, -x),
    ],
)
def test_reorient(item, root_player, target_player, expected):
    val = reorient(item, root_player=root_player, target_player=target_player)

    assert val == expected
