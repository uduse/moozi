# %%
from functools import partial

import moozi as mz
import ray
from absl import logging
from moozi.logging import JAXBoardStepData, MetricsReporterActor
from moozi.parameter_optimizer import ParameterOptimizer
from moozi.replay import ReplayBuffer
from moozi.utils import WallTimer

from utils import *

ray.init(ignore_reinit_error=True)


# %%
config = mz.Config().update(
    env=f"catch(columns=6,rows=6)",
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
param_opt = ray.remote(ParameterOptimizer).options(num_gpus=0.5).remote()

ray.get(
    [
        param_opt.build.remote(partial(make_param_opt_properties, config=config)),
        param_opt.build_loggers.remote(
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
def build_worker(config, param_opt):
    worker = ray.remote(RolloutWorkerWithWeights).remote()
    worker.set_network.remote(param_opt.get_network.remote())
    worker.set_params.remote(param_opt.get_params.remote())
    worker.build_batching_layers.remote(
        partial(make_rollout_worker_batching_layers, config=config)
    )
    worker.build_universes.remote(partial(make_rollout_worker_universes, config=config))
    return worker


rollout_workers = [
    build_worker(config, param_opt) for _ in range(config.num_rollout_workers)
]

# %%
evaluator = ray.remote(RolloutWorkerWithWeights).options(name="Evaluator").remote()
evaluator.set_network.remote(param_opt.get_network.remote())
evaluator.set_params.remote(param_opt.get_params.remote())
evaluator.build_universes.remote(partial(make_evaluator_universes, config=config))


@ray.remote
def evaluation_post_process(output_buffer):
    return JAXBoardStepData(
        scalars=dict(last_run_avr_reward=np.mean(output_buffer)), histograms=dict()
    )


# sanity check
# %% 
ray.get([w.run.remote(config.num_ticks_per_epoch) for w in rollout_workers])

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
