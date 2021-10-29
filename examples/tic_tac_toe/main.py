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
    env=f"tic_tac_toe",
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
    num_updates_per_samples_added=10,
    num_rollout_workers=1,
    num_rollout_universes_per_worker=1,
)

num_interactions = (
    config.num_epochs
    * config.num_ticks_per_epoch
    * config.num_rollout_workers
    * config.num_rollout_universes_per_worker
)
print(f"num_interactions: {num_interactions}")
config.print()

# %%
param_opt = ParameterOptimizer()
param_opt.build(partial(make_param_opt_properties, config=config)),
param_opt.build_loggers(
    lambda: [
        # TerminalLogger(label="Parameter Optimizer", print_fn=print),
        mz.logging.JAXBoardLogger(name="param_opt"),
    ]
),
param_opt.log_stats(),

# %%
replay_buffer = ray.remote(ReplayBuffer).remote(config)

# %%
def make_rollout_worker_universes(
    self: RolloutWorkerWithWeights, config: Config
) -> List[UniverseAsync]:
    dim_actions = make_env_spec(config.env).actions.num_values

    def make_rollout_universe(index):
        tape = Tape(index)
        planner_law = make_async_planner_law(
            lambda x: self.init_inf_fn_unbatched(self.params, x),
            lambda x: self.recurr_inf_fn_unbatched(self.params, x[0], x[1]),
            dim_actions=dim_actions,
        )
        laws = [
            EnvironmentLaw(make_env(config.env)),
            FrameStacker(num_frames=config.num_stacked_frames, player=0),
            set_policy_feed,
            planner_law,
            TrajectoryOutputWriter(),
            update_episode_stats,
            increment_tick,
        ]
        return UniverseAsync(tape, laws)

    universes = [
        make_rollout_universe(i) for i in range(config.num_rollout_universes_per_worker)
    ]
    return universes


# %%
worker = RolloutWorkerWithWeights()
worker.set_network(param_opt.get_network())
worker.set_params(param_opt.get_params())
worker.build_universes(partial(make_rollout_worker_universes, config=config))


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
