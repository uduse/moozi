from typing import AsyncContextManager, Awaitable, Callable, ContextManager, Coroutine
from absl import logging
import numpy as np
import contextlib
import attr
import trio

logging.set_verbosity(logging.INFO)


@attr.s(auto_attribs=True)
class BatchingClient:
    client_id: int
    send_request: Callable[..., Awaitable]
    receive_response: Callable[..., Awaitable]
    open_context: AsyncContextManager


@attr.s
class BatchingLayer:
    SEND = 0
    RECEIVE = 1

    max_batch_size = attr.ib()
    batch_process_period = attr.ib(default=1)

    batch_buffer = attr.ib(factory=list)
    batch_process_fn = attr.ib(default=None)

    request_channel = attr.ib(factory=lambda: trio.open_memory_channel(0))
    response_channels = attr.ib(factory=dict)

    stack = attr.ib(factory=contextlib.AsyncExitStack)
    client_counter = attr.ib(default=0)

    def get_client_id(self):
        client_id = self.client_counter
        self.client_counter += 1
        return client_id

    def spawn_client(self):
        client_id = self.get_client_id()

        self.response_channels[client_id] = trio.open_memory_channel(0)
        send_request_channel = self.request_channel[self.SEND].clone()
        receive_response_channel = self.response_channels[client_id][self.RECEIVE]

        async def send_request_fn(payload):
            return await send_request_channel.send(
                dict(client_id=client_id, payload=payload)
            )

        async def receive_response_fn():
            result = await receive_response_channel.receive()
            logging.debug(f"response for {client_id} received")
            return result

        @contextlib.asynccontextmanager
        async def open_context():
            yield
            async with trio.open_nursery() as n:
                n.start_soon(send_request_channel.aclose)
                n.start_soon(receive_response_channel.aclose)

        return BatchingClient(
            client_id, send_request_fn, receive_response_fn, open_context
        )

    async def start_processing(self):
        while True:
            try:
                with trio.move_on_after(self.batch_process_period):
                    while True:
                        data = await self.request_channel[self.RECEIVE].receive()
                        logging.debug(f"request from {data['client_id']} received")
                        self.batch_buffer.append(data)
                        if len(self.batch_buffer) >= self.max_batch_size:
                            break
                await self.process_batch()
            except (trio.EndOfChannel, trio.ClosedResourceError):
                assert len(self.batch_buffer) == 0
                break

    async def process_batch(self):
        if len(self.batch_buffer) > 0:
            client_ids = [d["client_id"] for d in self.batch_buffer]
            data = [d["payload"] for d in self.batch_buffer]
            processed_data = await self.batch_process_fn(data)
            logging.debug(
                f"{len(processed_data)} data processed, sending back to {client_ids}"
            )
            for client_id, d in zip(client_ids, processed_data):
                await self.response_channels[client_id][self.SEND].send(d)
            self.batch_buffer.clear()
        else:
            logging.debug("batch buffer empty, skipped")

    @contextlib.asynccontextmanager
    async def open_context(self):
        yield
        logging.debug("closing batching layer")
        async with trio.open_nursery() as n:
            n.start_soon(self.request_channel[self.RECEIVE].aclose)
            n.start_soon(self.request_channel[self.SEND].aclose)
            for client_id in self.response_channels:
                n.start_soon(self.response_channels[client_id][self.SEND].aclose)
        logging.debug("batching layer closed")
