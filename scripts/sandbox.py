# %%
import os

import tree

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from functools import partial
from typing import List

import moozi as mz
import numpy as np
import ray
from absl import logging
from moozi.core import Config, UniverseAsync, link
from moozi.core.env import make_env
from moozi.laws import (
    EnvironmentLaw,
    FrameStacker,
    TrajectoryOutputWriter,
    make_policy_feed,
    output_last_step_reward,
    update_episode_stats,
)
from moozi.logging import (
    JAXBoardLoggerActor,
    JAXBoardLoggerV2,
    JAXBoardStepData,
    LoggerDatumScalar,
)
from moozi.parameter_optimizer import ParameterOptimizer
from moozi.policy.mcts_async import ActionSamplerLaw, planner_law
from moozi.replay import ReplayBuffer
from moozi.rollout_worker import RolloutWorkerWithWeights
from moozi.utils import WallTimer

# from utils import *

print(ray.init(include_dashboard=True))


# %%
config = Config()
config.env = f"catch(columns=6,rows=6)"
config.batch_size = 256
config.discount = 0.99
config.num_unroll_steps = 3
config.num_td_steps = 100
config.num_stacked_frames = 1
config.lr = 1e-3
config.replay_buffer_size = 100000
config.num_epochs = 150
config.num_ticks_per_epoch = 12
config.num_updates_per_samples_added = 10
config.num_rollout_workers = 5
config.num_rollout_universes_per_worker = 20
config.weight_decay = 5e-3
config.nn_arch_cls = mz.nn.ResNetArchitecture

env_spec = mz.make_env_spec(config.env)
frame_shape = env_spec.observations.observation.shape
stacked_frames_shape = frame_shape[:-1] + (frame_shape[-1] * config.num_stacked_frames,)
dim_action = env_spec.actions.num_values
dim_repr = 16
config.nn_spec = mz.nn.ResNetSpec(
    stacked_frames_shape=stacked_frames_shape,
    dim_repr=dim_repr,
    dim_action=dim_action,
    repr_tower_blocks=6,
    repr_tower_dim=16,
    pred_tower_blocks=6,
    pred_tower_dim=16,
    dyna_tower_blocks=6,
    dyna_tower_dim=16,
    dyna_state_blocks=6,
)
config.print()

num_interactions = (
    config.num_epochs
    * config.num_ticks_per_epoch
    * config.num_rollout_workers
    * config.num_rollout_universes_per_worker
)
print(f"num_interactions: {num_interactions}")


#  %%
def make_parameter_optimizer():
    param_opt = ray.remote(num_cpus=1, num_gpus=1)(ParameterOptimizer).remote()
    param_opt.make_training_suite.remote(config)
    param_opt.make_loggers.remote(
        lambda: [
            # TODO: use new logger instead
            mz.logging.JAXBoardLogger(name="param_opt", time_delta=0),
        ]
    ),
    return param_opt


def make_laws_train(config):
    return [
        EnvironmentLaw(make_env(config.env), num_players=1),
        FrameStacker(num_frames=config.num_stacked_frames, player=0),
        make_policy_feed,
        planner_law,
        ActionSamplerLaw(),
        TrajectoryOutputWriter(),
    ]


def make_laws_eval(config):
    return [
        link(lambda: dict(num_simulations=30)),
        EnvironmentLaw(make_env(config.env), num_players=1),
        FrameStacker(num_frames=config.num_stacked_frames, player=0),
        make_policy_feed,
        planner_law,
        ActionSamplerLaw(temperature=0.1),
        output_last_step_reward,
    ]


def make_workers_train(config: Config, param_opt):
    workers = []
    for _ in range(config.num_rollout_workers):
        worker = ray.remote(RolloutWorkerWithWeights).remote()
        worker.set_model.remote(param_opt.get_model.remote())
        worker.set_params_and_state.remote(param_opt.get_params_and_state.remote())
        worker.make_batching_layers.remote(config)
        worker.make_universes_from_laws.remote(
            partial(make_laws_train, config), config.num_rollout_universes_per_worker
        )
        workers.append(worker)
    return workers


def make_worker_eval(config: Config, param_opt):
    worker = ray.remote(RolloutWorkerWithWeights).remote()
    worker.set_model.remote(param_opt.get_model.remote())
    worker.set_params_and_state.remote(param_opt.get_params_and_state.remote())
    worker.make_universes_from_laws.remote(partial(make_laws_eval, config), 1)
    return worker


@ray.remote
def convert_reward_to_logger_datum(rewards):
    return LoggerDatumScalar("mean_reward", np.mean(rewards))


# %%
param_opt = make_parameter_optimizer()
replay_buffer = ray.remote(ReplayBuffer).remote(config)
workers_train = make_workers_train(config, param_opt)
worker_eval = make_worker_eval(config, param_opt)
reporter = JAXBoardLoggerActor.remote()
# %%
def avr_g(worker: RolloutWorkerWithWeights):
    logging.info(
        f"R: {np.mean([u.tape.avg_episodic_reward for u in worker.universes])}"
    )


# %%
with WallTimer():
    for epoch in range(config.num_epochs):
        logging.info(f"Epochs: {epoch + 1} / {config.num_epochs}")
        samples = [w.run.remote(config.num_ticks_per_epoch) for w in workers_train]
        eval_result = worker_eval.run.remote(30)
        eval_result_logger_datum = convert_reward_to_logger_datum.remote(eval_result)
        samples_added = [replay_buffer.add_samples.remote(s) for s in samples]
        while samples_added:
            _, samples_added = ray.wait(samples_added)
            for _ in range(config.num_updates_per_samples_added):
                batch = replay_buffer.get_batch.remote(config.batch_size)
                param_opt.update.remote(batch)

            for w in workers_train + [worker_eval]:
                w.set_params_and_state.remote(param_opt.get_params_and_state.remote())

            param_opt.log.remote()
        reporter.write.remote(replay_buffer.get_logger_data.remote())
        done = reporter.write.remote(eval_result_logger_datum)

    ray.get(done)
