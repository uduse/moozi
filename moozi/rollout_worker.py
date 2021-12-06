import os
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, List

import jax
import ray
import trio
import trio_asyncio
from absl import logging

import moozi as mz
from moozi.core import UniverseAsync, link
from moozi.batching_layer import BatchingClient, BatchingLayer
from moozi.replay import StepSample, TrajectorySample, make_target_from_traj


@dataclass(repr=False)
class RolloutWorkerWithWeights:
    batching_layers: List[BatchingLayer] = field(default_factory=list)
    universes: List[UniverseAsync] = field(init=False)

    network: mz.nn.NeuralNetwork = field(init=False)
    params: Any = field(init=False)

    init_inf_fn: Callable = field(init=False)
    recurr_inf_fn: Callable = field(init=False)

    def build_batching_layers(self, factory):
        self.batching_layers = factory(self=self)

    def build_universes(self, factory):
        self.universes = factory(self=self)

    def set_verbosity(self, verbosity):
        logging.set_verbosity(verbosity)

    def run(self, num_ticks):
        async def main_loop():
            async with trio.open_nursery() as main_nursery:
                for b in self.batching_layers:
                    b.is_paused = False
                    main_nursery.start_soon(b.start_processing)
                    # TODO: toggle logging
                    # main_nursery.start_soon(b.start_logging)

                async with trio.open_nursery() as universe_nursery:
                    for u in self.universes:
                        universe_nursery.start_soon(partial(u.tick, times=num_ticks))

                for b in self.batching_layers:
                    b.is_paused = True

        trio_asyncio.run(main_loop)

        return self._flush_output_buffers()

    def _flush_output_buffers(self) -> List[TrajectorySample]:
        outputs: List[TrajectorySample] = sum(
            (list(u.tape.output_buffer) for u in self.universes), []
        )

        for u in self.universes:
            u.tape.output_buffer = tuple()
        return outputs

    def set_params(self, params):
        self.params = params

    def set_network(self, network: mz.nn.NeuralNetwork):
        # logging.info("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
        # logging.info(
        #     f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', None)}"
        # )
        # logging.info(f"jax.devices(): {jax.devices()}")
        self.network = network

        self.init_inf_fn = jax.jit(network.initial_inference)
        self.recurr_inf_fn = jax.jit(network.recurrent_inference)
        self.init_inf_fn_unbatched = jax.jit(network.initial_inference_unbatched)
        self.recurr_inf_fn_unbatched = jax.jit(network.recurrent_inference_unbatched)

    def exec(self, fn):
        return fn(self)


# @dataclass(repr=False)
# class RolloutWorkerV1:
#     batching_layers: Optional[List[BatchingLayer]] = field(default_factory=list)
#     universes: List[UniverseAsync] = field(init=False)

#     def make_batching_layers(self, factory):
#         self.batching_layers = factory(self)

#     def make_universes(self, factory):
#         self.universes = factory(self)

#     def set_verbosity(self, verbosity):
#         logging.set_verbosity(verbosity)

#     def run(self, num_ticks):
#         async def main_loop():
#             async with trio.open_nursery() as main_nursery:
#                 for b in self.batching_layers:
#                     b.is_paused = False
#                     main_nursery.start_soon(b.start_processing)
#                     # TODO: toggle logging
#                     # main_nursery.start_soon(b.start_logging)

#                 async with trio.open_nursery() as universe_nursery:
#                     for u in self.universes:
#                         universe_nursery.start_soon(partial(u.tick, times=num_ticks))

#                 for b in self.batching_layers:
#                     b.is_paused = True

#         trio_asyncio.run(main_loop)

#         return self._flush_output_buffers()

#     def _flush_output_buffers(self) -> List[TrajectorySample]:
#         outputs: List[TrajectorySample] = sum(
#             (list(u.artifact.output_buffer) for u in self.universes), []
#         )

#         for u in self.universes:
#             u.artifact.output_buffer = tuple()
#         return outputs

#     def set_params(self, params):
#         self.params = params

#     def set_network(self, network: mz.nn.NeuralNetwork):
#         logging.info("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
#         logging.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
#         logging.info(f"jax.devices(): {jax.devices()}")
#         self.network = network

#         self.init_inf_fn = jax.jit(network.initial_inference, backend="cpu")
#         self.recurr_inf_fn = jax.jit(network.recurrent_inference, backend="cpu")
#         self.init_inf_fn_unbatched = jax.jit(
#             network.initial_inference_unbatched, backend="cpu"
#         )
#         self.recurr_inf_fn_unbatched = jax.jit(
#             network.recurrent_inference_unbatched, backend="cpu"
#         )

#     def exec(self, fn):
#         return fn(self)
