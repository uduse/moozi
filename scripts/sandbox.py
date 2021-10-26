# %%
from functools import partial

import chex
import dm_env
import jax
import jax.numpy as jnp
import moozi as mz
import numpy as np
import optax
import ray
import tree
from absl import logging
from acme import specs
from acme.utils.loggers import TerminalLogger
from acme.utils.tree_utils import stack_sequence_fields, unstack_sequence_fields

from acme.wrappers import OpenSpielWrapper, SinglePrecisionWrapper, open_spiel_wrapper
from acme.wrappers.frame_stacking import FrameStacker
from moozi import batching_layer
from moozi.batching_layer import BatchingClient, BatchingLayer
from moozi.config import Config
from moozi.laws import *
from moozi.link import Universe, UniverseAsync, link
from moozi.logging import JAXBoardLogger, JAXBoardStepData
from moozi.materia import Materia
from moozi.nn import NeuralNetwork
from moozi.policy.mcts_async import MCTSAsync, make_async_planner_law
from moozi.policy.policy import PolicyFeed
from moozi.replay import StepSample, TrajectorySample, make_target_from_traj
from moozi.rollout_worker import RolloutWorkerWithWeights
from moozi.utils import SimpleBuffer, WallTimer, as_coroutine, check_ray_gpu
from trio_asyncio import aio_as_trio

logging.set_verbosity(logging.INFO)


ray.init(ignore_reinit_error=True)


# %%
config = Config().update(
    batch_size=256,
    discount=0.99,
    num_unroll_steps=3,
    num_td_steps=100,
    num_stacked_frames=1,
    lr=2e-3,
    replay_buffer_size=10000,
    dim_repr=64,
    num_epochs=20,
    num_ticks_per_epoch=10,
    num_updates_per_samples_added=30,
    num_rollout_workers=8,
    num_rollout_universes_per_worker=100,
)

num_interactions = (
    config.num_epochs
    * config.num_ticks_per_epoch
    * config.num_rollout_workers
    * config.num_rollout_universes_per_worker
)
print(f"num_interactions: {num_interactions}")

# %%
metrics_reporter = MetricsReporterActor.remote()
# %%
param_opt = ray.remote(ParameterOptimizer).options(num_gpus=0.5).remote()

ray.get(
    [
        param_opt.setup.remote(partial(setup_param_opt, config=config)),
        param_opt.set_loggers.remote(
            lambda: [
                # TerminalLogger(label="Parameter Optimizer", print_fn=print),
                mz.logging.JAXBoardLogger(name="param_opt"),
            ]
        ),
        param_opt.log_stats.remote(),
    ]
)

# %%
replay_buffer = ray.remote(ReplayBuffer).remote(config)

# %%
rollout_workers = []
for _ in range(config.num_rollout_workers):
    worker = ray.remote(RolloutWorker).remote()
    worker.set_network.remote(param_opt.get_network.remote())
    worker.set_params.remote(param_opt.get_params.remote())
    worker.set_batching_layers.remote(make_rollout_worker_batching_layers)
    worker.set_universes.remote(partial(make_rollout_worker_universes, config=config))
    rollout_workers.append(worker)

# %%
# ray.get(worker.run.remote(10))


# %%
evaluator = ray.remote(RolloutWorker).options(name="Evaluator").remote()
evaluator.set_network.remote(param_opt.get_network.remote())
evaluator.set_params.remote(param_opt.get_params.remote())
evaluator.set_universes.remote(partial(make_evaluator_universes, config=config))


@ray.remote
def evaluation_post_process(output_buffer):
    return JAXBoardStepData(
        scalars=dict(last_run_avr_reward=np.mean(output_buffer)), histograms=dict()
    )


# %%
def evaluate(num_ticks):
    output_buffer = evaluator.run.remote(num_ticks)
    step_data = evaluation_post_process.remote(output_buffer)
    return metrics_reporter.report.remote(step_data)


# %%
@ray.remote
def log(result):
    logging.info(result)


# %%
with WallTimer():
    for epoch in range(config.num_epochs):
        logging.info(f"Epochs: {epoch + 1} / {config.num_epochs}")

        evaluation_done = evaluate(num_ticks=50)
        samples = [w.run.remote(config.num_ticks_per_epoch) for w in rollout_workers]
        samples_added = [replay_buffer.add_samples.remote(s) for s in samples]
        while samples_added:
            _, samples_added = ray.wait(samples_added)
            for _ in range(config.num_updates_per_samples_added):
                batch = replay_buffer.get_batch.remote(config.batch_size)
                param_opt.update.remote(batch)

            for w in rollout_workers + [evaluator]:
                w.set_params.remote(param_opt.get_params.remote())

            param_opt.log_stats.remote()

    ray.get(evaluation_done)
