import inspect
from dataclasses import dataclass, field
from typing import (
    AsyncContextManager,
    Awaitable,
    Callable,
    ClassVar,
    ContextManager,
    Coroutine,
)
from absl import logging
import numpy as np
import contextlib
import attr
import trio

logging.set_verbosity(logging.INFO)


# @attr.s(auto_attribs=True, repr=False)
@dataclass
class BatchingClient:
    client_id: int
    # send_request: Callable[..., Awaitable]
    # receive_response: Callable[..., Awaitable]
    request: Callable[..., Awaitable]
    open_context: AsyncContextManager


# @attr.s(repr=False)
@dataclass
class BatchingLayer:
    SEND: ClassVar[int] = 0
    RECEIVE: ClassVar[int] = 1

    max_batch_size: int
    batch_process_period: float = 1

    batch_buffer: list = field(default_factory=list)
    process_fn: Callable = lambda: None

    request_channel: list = field(default_factory=lambda: trio.open_memory_channel(0))
    response_channels: dict = field(default_factory=dict)

    client_counter: int = 0

    def get_client_id(self):
        client_id = self.client_counter
        self.client_counter += 1
        return client_id

    def spawn_client(self):
        client_id = self.get_client_id()

        self.response_channels[client_id] = trio.open_memory_channel(0)
        send_request_channel = self.request_channel[self.SEND].clone()
        receive_response_channel = self.response_channels[client_id][self.RECEIVE]

        async def request_fn(payload):
            await send_request_channel.send(dict(client_id=client_id, payload=payload))
            response = await receive_response_channel.receive()
            # logging.debug(f"response for {client_id} received")
            return response

        @contextlib.asynccontextmanager
        async def open_context():
            yield
            async with trio.open_nursery() as n:
                n.start_soon(send_request_channel.aclose)
                n.start_soon(receive_response_channel.aclose)

        return BatchingClient(client_id, request_fn, open_context)

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
            processed_data = self.process_fn(data)
            if inspect.isawaitable(processed_data):
                processed_data = await processed_data
            logging.debug(
                f"{len(processed_data)} data processed, sending back to {len(client_ids)} clients"
            )
            for client_id, d in zip(client_ids, processed_data):
                await self.response_channels[client_id][self.SEND].send(d)
            self.batch_buffer.clear()
        else:
            logging.debug("batch buffer empty, skipped")

    async def close(self):
        logging.debug("closing batching layer")
        async with trio.open_nursery() as n:
            n.start_soon(self.request_channel[self.RECEIVE].aclose)
            n.start_soon(self.request_channel[self.SEND].aclose)
            for client_id in self.response_channels:
                n.start_soon(self.response_channels[client_id][self.SEND].aclose)
        logging.debug("batching layer closed")
