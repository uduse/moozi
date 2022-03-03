import os
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, List, Optional

import haiku as hk
import jax
import numpy as np
import ray
import tree
import trio
import trio_asyncio
from absl import logging
from acme.utils.tree_utils import stack_sequence_fields, unstack_sequence_fields

import moozi as mz
from moozi.batching_layer import BatchingClient, BatchingLayer
from moozi.core import Config, Tape, UniverseAsync, link
from moozi.nn.nn import RootFeatures, TransitionFeatures
from moozi.replay import StepSample, TrajectorySample, make_target_from_traj


@dataclass(repr=False)
class RolloutWorkerWithWeights:
    universes: List[UniverseAsync] = field(init=False)
    model: mz.nn.NNModel = field(init=False)
    params: hk.Params = field(init=False)
    state: hk.State = field(init=False)

    batching_layers: List[BatchingLayer] = field(default_factory=list)

    def make_batching_layers(self, config: Config):
        def batched_root_inf(feats: List[RootFeatures]):
            batch_size = len(feats)
            nn_outputs, _ = self.model.root_inference(
                self.params, self.state, stack_sequence_fields(feats), is_training=False
            )
            nn_outputs = tree.map_structure(np.array, nn_outputs)
            return unstack_sequence_fields(nn_outputs, batch_size)

        def batched_trans_inf(feats: List[TransitionFeatures]):
            batch_size = len(feats)
            nn_outputs, _ = self.model.trans_inference(
                self.params, self.state, stack_sequence_fields(feats), is_training=False
            )
            nn_outputs = tree.map_structure(np.array, nn_outputs)
            return unstack_sequence_fields(nn_outputs, batch_size)

        bl_root_inf = BatchingLayer(
            max_batch_size=config.num_rollout_universes_per_worker,
            process_fn=batched_root_inf,
            name="[batched_root_inf]",
            batch_process_period=1e-1,
        )
        bl_trans_inf = BatchingLayer(
            max_batch_size=config.num_rollout_universes_per_worker,
            process_fn=batched_trans_inf,
            name="[batche_trans_inf]",
            batch_process_period=1e-1,
        )
        self.batching_layers = [bl_root_inf, bl_trans_inf]

    def make_universes_from_laws(self, laws_factory, num_universes):
        def _make_universe(index):
            tape = Tape(index)
            laws = laws_factory()
            return UniverseAsync(tape, laws)

        self.universes = [_make_universe(i) for i in range(num_universes)]

    def _set_inferences(self):
        # TODO: check client unnecsesary respawned
        if self.batching_layers:
            for u in self.universes:
                u.tape.root_inf_fn = self.batching_layers[0].spawn_client().request
                u.tape.trans_inf_fn = self.batching_layers[1].spawn_client().request
        else:
            for u in self.universes:

                async def root_inf(feats):
                    return self.model.root_inference_unbatched(
                        self.params, self.state, feats, is_training=False
                    )[0]

                u.tape.root_inf_fn = root_inf

                async def trans_inf(feats):
                    return self.model.trans_inference_unbatched(
                        self.params, self.state, feats, is_training=False
                    )[0]

                u.tape.trans_inf_fn = trans_inf

    def set_verbosity(self, verbosity):
        logging.set_verbosity(verbosity)

    def run(self, num_ticks):
        async def main_loop():
            async with trio.open_nursery() as main_nursery:
                self._set_inferences()

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

    def set_params_and_state(self, params_and_state):
        self.params, self.state = params_and_state

    def set_model(self, model: mz.nn.NNModel):
        self.model = model.with_jit()

    def exec(self, fn):
        return fn(self)
