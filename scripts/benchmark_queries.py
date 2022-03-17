# %%
import collections
import time

import acme
import dm_env
import jax
from jax._src.dtypes import dtype
import moozi as mz
import numpy as np
import open_spiel
import ray
import ray.util.queue
import tree
from acme.agents.jax.impala.types import Observation
from acme.jax.networks.base import Value
from acme.wrappers import SinglePrecisionWrapper
from acme.wrappers.open_spiel_wrapper import OpenSpielWrapper
from moozi.utils import SimpleBuffer
import pandas as pd

ray.init(ignore_reinit_error=True)

# %%
raw_env = open_spiel.python.rl_environment.Environment(f"tic_tac_toe")
env = acme.wrappers.open_spiel_wrapper.OpenSpielWrapper(raw_env)
env = acme.wrappers.SinglePrecisionWrapper(env)
env_spec = acme.specs.make_environment_spec(env)

# %%
num_stacked_frames = 2
dim_repr = 64
dim_action = env_spec.actions.num_values
frame_shape = env_spec.observations.observation.shape
stacked_frame_shape = (num_stacked_frames,) + frame_shape
nn_spec = mz.nn.NNSpec(
    obs_rows=stacked_frame_shape,
    dim_repr=dim_repr,
    dim_action=dim_action,
    repr_net_sizes=(128, 128),
    pred_net_sizes=(128, 128),
    dyna_net_sizes=(128, 128),
)
network = mz.nn.get_network(nn_spec)
master_key = jax.random.PRNGKey(0)
params = network.init_network(master_key)

# %%
@ray.remote
def requester(requester_id, request_q):
    response_q = ray.util.queue.Queue(maxsize=1)

    counter = 0
    while True:
        request_q.put((response_q, f"{requester_id}/{counter}"))
        response = response_q.get()
        print(response, "received")
        time.sleep(0.1)
        counter += 1
        if counter >= 100:
            break


@ray.remote
class Learner(object):
    def __init__(self, queue: ray.util.queue.Queue, network):
        master_key = jax.random.PRNGKey(0)
        self._queue = queue
        self._network = network
        # self._params = network.init(master_key)
        self._batch_size = 8

    def loop(self):
        while True:
            if self._queue.size() >= self._batch_size:
                items = self._queue.get_nowait_batch(self._batch_size)
                for response_q, counter in items:
                    response_q.put("I saw " + str(counter))


# %%
class WallTimer(object):
    def __init__(self):
        self._enter_time = 0
        self._time_sum = 0

    def __enter__(self):
        self._enter_time = time.time()

    def __exit__(self, *args):
        time_diff = time.time() - self._enter_time
        self._time_sum += time_diff

    @property
    def time_sum(self):
        return self._time_sum

    def print(self):
        print(f"Time Elpased: {str(self.time_sum)}")


# %%
request_q = ray.util.queue.Queue()
requestors = [requester.remote(i, request_q) for i in range(20)]

# %%
learner = Learner.remote(request_q, None)
learner.loop.remote()

# %%
import collections
import random

num_experiments = 5
num_inputs = 10000
results = collections.defaultdict(list)

init_inf_fn = jax.jit(network.root_inference, backend="cpu")
recurr_inf_fn = jax.jit(network.trans_inference, backend="cpu")
num_recurr_inf = 10

for _ in range(num_experiments):
    inputs = [np.random.randn(*stacked_frame_shape) for _ in range(num_inputs)]

    num_batches_pool = [2 ** i for i in range(14)]
    random.shuffle(num_batches_pool)

    for num_batches in num_batches_pool:
        inputs_batched = np.array_split(np.array(inputs), num_batches)
        actions_batched = [
            np.zeros(inputs_batched[i].shape[0], dtype=int)
            for i in range(len(inputs_batched))
        ]

        t = WallTimer()
        with t:
            for batch, actions in zip(inputs_batched, actions_batched):
                network_output = init_inf_fn(params, batch)
                for _ in range(num_recurr_inf):
                    network_output = recurr_inf_fn(
                        params, network_output.hidden_state, actions
                    )

        key = f"batch_size: {inputs_batched[0].shape[0]}, num_batches: {len(inputs_batched)}"
        results[num_batches].append(t.time_sum)

# %%
pd.DataFrame(pd.DataFrame(results).mean().sort_values())
