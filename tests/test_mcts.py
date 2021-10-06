from moozi.batching_layer import BatchingClient, BatchingLayer
from moozi.nn import NeuralNetwork
from moozi.policies.mcts_async import MCTSAsync


async def test_async_mcts(network: NeuralNetwork, params):
    # async def process_fn(batch):
    #     return network.initial_inference(batch)

    # batching_layer = BatchingLayer(max_batch_size=5, process_fn=process_fn)
    assert MCTSAsync
    

