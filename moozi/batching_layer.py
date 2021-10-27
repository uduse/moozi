import inspect
from dataclasses import dataclass, field
from typing import (
    AsyncContextManager,
    Awaitable,
    Callable,
    ClassVar
)
from absl import logging
import contextlib
import trio

logging.set_verbosity(logging.INFO)


# TODO: rewrite everything in asyncio?


@dataclass
class BatchingClient:
    client_id: int
    request: Callable[..., Awaitable]
    open_context: AsyncContextManager


@dataclass
class BatchingLayer:
    SEND: ClassVar[int] = 0
    RECEIVE: ClassVar[int] = 1

    max_batch_size: int
    name: str = "BatchingLayer"
    batch_process_period: float = 1

    batch_buffer: list = field(default_factory=list)
    process_fn: Callable = lambda x: x

    request_channel: list = field(default_factory=lambda: trio.open_memory_channel(0))
    response_channels: dict = field(default_factory=dict)

    is_paused: bool = False

    logging_period: float = 5
    logging_throughput: int = 0

    _client_counter: int = 0

    def _get_client_id(self):
        client_id = self._client_counter
        self._client_counter += 1
        return client_id

    def spawn_client(self):
        client_id = self._get_client_id()

        self.response_channels[client_id] = trio.open_memory_channel(0)
        send_request_channel = self.request_channel[self.SEND].clone()
        receive_response_channel = self.response_channels[client_id][self.RECEIVE]

        async def request_fn(payload):
            await send_request_channel.send(dict(client_id=client_id, payload=payload))
            response = await receive_response_channel.receive()
            return response

        @contextlib.asynccontextmanager
        async def open_context():
            yield
            async with trio.open_nursery() as n:
                n.start_soon(send_request_channel.aclose)
                n.start_soon(receive_response_channel.aclose)

        return BatchingClient(client_id, request_fn, open_context)

    async def start_processing(self):
        logging.debug(f"{self.name} started processing")
        while not self.is_paused:
            try:
                with trio.move_on_after(self.batch_process_period):
                    while True:
                        data = await self.request_channel[self.RECEIVE].receive()
                        # logging.debug(f"request from {data['client_id']} received")
                        self.batch_buffer.append(data)
                        if len(self.batch_buffer) >= self.max_batch_size:
                            break
                await self.process_batch()
            except (trio.EndOfChannel, trio.ClosedResourceError):
                # TODO: use a event to signal stop processing instead of
                #       closing the channels?
                assert len(self.batch_buffer) == 0
                break
        logging.debug(f"{self.name} paused processing")

    async def process_batch(self):
        if len(self.batch_buffer) > 0:
            client_ids = [d["client_id"] for d in self.batch_buffer]
            data = [d["payload"] for d in self.batch_buffer]
            processed_data = self.process_fn(data)
            if inspect.isawaitable(processed_data):
                processed_data = await processed_data
            assert len(processed_data) == len(client_ids)
            self.logging_throughput += len(processed_data)
            for client_id, d in zip(client_ids, processed_data):
                await self.response_channels[client_id][self.SEND].send(d)
            self.batch_buffer.clear()
        else:
            logging.debug(f"{self.name} empty, skipped")

    async def start_logging(self):
        # TODO: improve this start_logging interface
        while not self.is_paused:
            if self.logging_period > 0:
                await trio.sleep(self.logging_period)
                logging.debug(
                    f"{self.name} processed {self.logging_throughput} requests in {self.logging_period} second(s)"
                )
                self.logging_throughput = 0
        logging.debug(f"{self.name} processed {self.logging_throughput} requests.")
        self.logging_throughput = 0

    async def close(self):
        logging.debug(f"closing {self.name}")
        self.is_paused = True
        async with trio.open_nursery() as n:
            n.start_soon(self.request_channel[self.RECEIVE].aclose)
            n.start_soon(self.request_channel[self.SEND].aclose)
            for client_id in self.response_channels:
                n.start_soon(self.response_channels[client_id][self.SEND].aclose)
        logging.debug(f"{self.name} closed")

    @contextlib.asynccontextmanager
    async def open_context(self):
        yield
        await self.close()
