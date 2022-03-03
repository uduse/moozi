# %%
from typing import List
from functools import partial

import moozi as mz
import ray
from absl import logging
from moozi.core import UniverseAsync, Config
from moozi.core.env import make_env
from moozi.laws import (
    EnvironmentLaw,
    FrameStacker,
    TrajectoryOutputWriter,
    make_policy_feed,
)
from moozi.logging import JAXBoardStepData
from moozi.parameter_optimizer import ParameterOptimizer
from moozi.policy.mcts_async import ActionSamplerLaw, planner_law
from moozi.rollout_worker import RolloutWorkerWithWeights
from moozi.replay import ReplayBuffer
from moozi.utils import WallTimer

# from utils import *

ray.init(ignore_reinit_error=True)


# %%
config = Config()
config.env = f"catch(columns=7,rows=7)"
config.batch_size = 128
config.discount = 1
config.num_unroll_steps = 3
config.num_td_steps = 100
config.num_stacked_frames = 1
config.lr = 2e-3
config.replay_buffer_size = 10000
config.num_epochs = 20
config.num_ticks_per_epoch = 10
config.num_updates_per_samples_added = 30
config.num_rollout_workers = 1
config.num_rollout_universes_per_worker = 100
config.nn_arch_cls = mz.nn.ResNetArchitecture

env_spec = mz.make_env_spec(config.env)
frame_shape = env_spec.observations.observation.shape
stacked_frames_shape = frame_shape[:-1] + (frame_shape[-1] * config.num_stacked_frames,)
dim_action = env_spec.actions.num_values
dim_repr = 4
config.nn_spec = mz.nn.ResNetSpec(
    stacked_frames_shape=stacked_frames_shape, dim_repr=dim_repr, dim_action=dim_action
)
config.print()

num_interactions = (
    config.num_epochs
    * config.num_ticks_per_epoch
    * config.num_rollout_workers
    * config.num_rollout_universes_per_worker
)
print(f"num_interactions: {num_interactions}")


# %%
def make_parameter_optimizer():
    param_opt = ray.remote(num_cpus=1, num_gpus=1)(ParameterOptimizer).remote()
    param_opt.make_nn_suite.remote(config),
    param_opt.make_loggers.remote(
        lambda: [
            mz.logging.JAXBoardLoggerV2(name="param_opt", time_delta=0),
        ]
    ),
    return param_opt


# %%
def make_laws(config):
    return [
        EnvironmentLaw(make_env(config.env), num_players=1),
        FrameStacker(num_frames=config.num_stacked_frames, player=0),
        make_policy_feed,
        planner_law,
        ActionSamplerLaw(),
        TrajectoryOutputWriter(),
    ]


def make_workers_train(config: Config, param_opt):
    workers = []
    for _ in range(config.num_rollout_workers):
        worker = ray.remote(RolloutWorkerWithWeights).remote()
        worker.set_model.remote(param_opt.get_model.remote())
        worker.set_params_and_state.remote(param_opt.get_params_and_state.remote())
        worker.make_batching_layers.remote(config)
        worker.make_universes_from_laws.remote(
            partial(make_laws, config), config.num_rollout_universes_per_worker
        )
        workers.append(worker)
    return workers


# param_opt = make_parameter_optimizer()
# replay_buffer = ray.remote(ReplayBuffer).remote(config)
# workers_train = make_workers_train(config, param_opt)

# # sanity check
# ray.get([w.run.remote(config.num_ticks_per_epoch) for w in workers_train])

# # %%
# evaluator = ray.remote(RolloutWorkerWithWeights).options(name="Evaluator").remote()
# evaluator.set_network.remote(param_opt.get_network.remote())
# evaluator.set_params.remote(param_opt.get_params.remote())
# evaluator.make_universes.remote(partial(make_evaluator_universes, config=config))


# @ray.remote
# def evaluation_post_process(output_buffer):
#     return JAXBoardStepData(
#         scalars=dict(last_run_avr_reward=np.mean(output_buffer)), histograms=dict()
#     )


# # sanity check
# # %%
# ray.get([w.run.remote(config.num_ticks_per_epoch) for w in rollout_workers])

# # %%
# def evaluate(num_ticks):
#     output_buffer = evaluator.run.remote(num_ticks)
#     step_data = evaluation_post_process.remote(output_buffer)
#     return metrics_reporter.report.remote(step_data)


# # %%
# @ray.remote
# def log(result):
#     logging.info(result)


# # %%
# with WallTimer():
#     for epoch in range(config.num_epochs):
#         logging.info(f"Epochs: {epoch + 1} / {config.num_epochs}")

#         evaluation_done = evaluate(num_ticks=50)
#         samples = [w.run.remote(config.num_ticks_per_epoch) for w in rollout_workers]
#         samples_added = [replay_buffer.add_samples.remote(s) for s in samples]
#         while samples_added:
#             _, samples_added = ray.wait(samples_added)
#             for _ in range(config.num_updates_per_samples_added):
#                 batch = replay_buffer.get_batch.remote(config.batch_size)
#                 param_opt.update.remote(batch)

#             for w in rollout_workers + [evaluator]:
#                 w.set_params.remote(param_opt.get_params.remote())

#             param_opt.log_stats.remote()

#     ray.get(evaluation_done)

# %%
