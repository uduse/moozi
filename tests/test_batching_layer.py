from absl import logging
import numpy as np
import trio
from dataclasses import dataclass
from moozi.batching_layer import BatchingClient, BatchingLayer


@dataclass
class MockRequester:
    client: BatchingClient

    async def run(self):
        async with self.client.open_context():
            for _ in range(10):
                features = np.random.randn(1)
                logging.debug(f"{self.client.client_id} sending")
                result = await self.client.request(features)
                assert result == features + 1
        logging.info(f"requester {self.client.client_id} closed")


async def test_batching_layer():
    async def batch_processor_fn(data):
        return np.array(data) + 1

    batching_layer = BatchingLayer(
        max_batch_size=50, batch_process_period=0.1, process_fn=batch_processor_fn
    )
    clients = [batching_layer.spawn_client() for _ in range(10)]
    requesters = [MockRequester(client=c) for c in clients]

    async with trio.open_nursery() as nursery:
        nursery.start_soon(batching_layer.start_processing)

        async with trio.open_nursery() as requesters_nursery:
            for r in requesters:
                requesters_nursery.start_soon(r.run)
        logging.info("requesters done")

        await batching_layer.close()

    logging.info("all done")
