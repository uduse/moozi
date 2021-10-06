# %%
from typing import Any, Callable, Coroutine, Dict, List, NamedTuple, Optional, Union

import attr
import dm_env
import numpy as np
import ray
import trio_asyncio
from trio_asyncio import aio_as_trio
from absl import logging
from acme.wrappers import SinglePrecisionWrapper
from acme.wrappers.open_spiel_wrapper import OpenSpielWrapper
from jax._src.numpy.lax_numpy import stack
from moozi.batching_layer import BatchingClient, BatchingLayer
from moozi.link import AsyncUniverse, link
from moozi.utils import SimpleBuffer

from interactions import InteractionManager

logging.set_verbosity(logging.DEBUG)

ray.init(ignore_reinit_error=True)


# %%
def get_random_action_from_timestep(timestep: dm_env.TimeStep):
    if not timestep.last():
        legal_actions = timestep.observation[0].legal_actions
        random_action = np.random.choice(np.flatnonzero(legal_actions == 1))
        return {"action": random_action}
    else:
        return {"action": -1}


class PolicyServer:
    def __init__(self) -> None:
        logging.set_verbosity(logging.INFO)

    async def process_batch(self, batch):
        batch_len = len(batch)

        results = [get_random_action_from_timestep(timestep) for timestep in batch]
        assert len(results) == batch_len
        logging.debug(f"processed batch with len {batch_len}")
        return results


@attr.s
class Driver:
    def run(self):
        policy_server = ray.remote(PolicyServer).remote()

        batching_layer = BatchingLayer(
            max_batch_size=3,
            process_fn=lambda batch: aio_as_trio(
                policy_server.process_batch.remote(batch)
            ),
        )

        mgr = ray.remote(InteractionManager).remote(
            batching_layers=[batching_layer], num_universes=5, num_ticks=100
        )

        ref = mgr.run.remote()
        return ref
