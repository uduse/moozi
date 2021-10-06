from acme.jax.networks.base import NetworkOutput
from moozi.batching_layer import BatchingClient, BatchingLayer
from moozi.nn import NeuralNetwork
from moozi.policies.mcts_async import MCTSAsync
from moozi.policies.policy import PolicyFeed


async def test_async_mcts(network: NeuralNetwork, params, policy_feed: PolicyFeed):
    async def init_inf(frames):
        return network.initial_inference_unbatched(params, frames)

    def recurr_inf(inputs):
        return network.recurrent_inference_unbatched(params, *inputs)

    mcts_async = MCTSAsync(
        init_inf_fn=init_inf, recurr_inf_fn=recurr_inf, num_simulations=10, dim_action=2
    )

    results = mcts_async(policy_feed)
    print(results)
    assert results
