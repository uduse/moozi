# %%
from loguru import logger
import os
from functools import partial
from typing import List, NamedTuple

import haiku as hk
import jax
import moozi as mz
import numpy as np
import optax
import ray
import tree
from absl import logging
from moozi.core import Config, link
from moozi.core.env import make_env
from moozi.core.link import link
from moozi.core.tape import Tape
from moozi.laws import (
    AtariEnvWrapper,
    FrameStacker,
    OpenSpielEnvLaw,
    TrajectoryOutputWriter,
    make_policy_feed,
    output_last_step_reward,
    update_episode_stats,
)
from moozi.logging import (
    JAXBoardLoggerActor,
    JAXBoardLoggerV2,
    LogScalar,
    TerminalLogger,
    TerminalLoggerActor,
)
from moozi.nn.loss import MuZeroLoss
from moozi.nn.nn import RootFeatures, make_model
from moozi.nn.resnet import ResNetArchitecture, ResNetSpec
from moozi.parameter_optimizer import ParameterOptimizer
from moozi.policy import ActionSampler, planner_law
from moozi.replay import ReplayBuffer
from moozi.rollout_worker import RolloutWorkerWithWeights
from moozi.utils import WallTimer

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# %%
num_epochs = 50

config = Config()
config.env = "PongNoFrameskip-v4"
config.batch_size = 128
config.discount = 0.99
config.num_unroll_steps = 2
config.num_td_steps = 10
config.num_stacked_frames = 4
config.lr = 3e-3
config.replay_max_size = 100000
config.num_epochs = num_epochs
config.num_ticks_per_epoch = 1000
config.num_samples_per_update = 30
config.num_env_workers = 5
config.num_universes_per_env_worker = 5
config.weight_decay = 5e-2
config.nn_arch_cls = mz.nn.ResNetArchitecture
config.known_bound_min = -1
config.known_bound_max = 1

# %%
env_spec = mz.make_spec(config.env)
single_frame_shape = env_spec.observations.shape
obs_rows, obs_cols = single_frame_shape[:2]
obs_channels = single_frame_shape[-1] * config.num_stacked_frames
repr_rows = 6
repr_cols = 6
repr_channels = 8
dim_action = env_spec.actions.num_values

# %%
config.nn_spec = mz.nn.ResNetSpec(
    obs_rows=obs_rows,
    obs_cols=obs_cols,
    obs_channels=obs_channels,
    repr_rows=6,
    repr_cols=6,
    repr_channels=repr_channels,
    dim_action=dim_action,
    repr_tower_blocks=12,
    repr_tower_dim=12,
    pred_tower_blocks=12,
    pred_tower_dim=12,
    dyna_tower_blocks=12,
    dyna_tower_dim=12,
    dyna_state_blocks=12,
)
config.print()

num_interactions = (
    config.num_epochs
    * config.num_ticks_per_epoch
    * config.num_env_workers
    * config.num_universes_per_env_worker
)
print(f"num_interactions: {num_interactions}")

# %%
def make_parameter_optimizer(config):
    param_opt = ray.remote(num_cpus=1, num_gpus=0.4)(ParameterOptimizer).remote()
    param_opt.make_training_suite.remote(config)
    param_opt.make_loggers.remote(
        lambda: [
            mz.logging.JAXBoardLoggerV2(name="param_opt", time_delta=15),
        ]
    ),
    return param_opt


def make_laws_train(config):
    return [
        AtariEnvWrapper(make_env(config.env)),
        FrameStacker(num_frames=config.num_stacked_frames, player=0),
        make_policy_feed,
        planner_law,
        ActionSampler(),
        TrajectoryOutputWriter(),
    ]


def make_laws_eval(config):
    return [
        link(lambda: dict(num_simulations=15)),
        AtariEnvWrapper(make_env(config.env), record_video=True),
        FrameStacker(num_frames=config.num_stacked_frames, player=0),
        make_policy_feed,
        planner_law,
        ActionSampler(temperature=0.1),
        output_last_step_reward,
    ]


def make_workers_train(config: Config, param_opt):
    workers = []
    for _ in range(config.num_env_workers):
        worker = ray.remote(RolloutWorkerWithWeights).remote()
        worker.set_model.remote(param_opt.get_model.remote())
        worker.set_params_and_state.remote(param_opt.get_params_and_state.remote())
        worker.make_batching_layers.remote(config)
        worker.make_universes_from_laws.remote(
            partial(make_laws_train, config), config.num_universes_per_env_worker
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
    return LogScalar("mean_reward", np.mean(rewards))


# %%
param_opt = make_parameter_optimizer(config)
replay_buffer = ray.remote(ReplayBuffer).remote(config)
workers_train = make_workers_train(config, param_opt)
worker_eval = make_worker_eval(config, param_opt)
jaxboard_logger = JAXBoardLoggerActor.remote("jaxboard_logger")
terminal_logger = TerminalLoggerActor.remote("terminal_logger")
terminal_logger.write.remote(param_opt.get_properties.remote())

# %%
with WallTimer():
    for epoch in range(config.num_epochs):
        logger.info(f"Epochs: {epoch + 1} / {config.num_epochs}")
        samples = [w.run.remote(config.num_ticks_per_epoch) for w in workers_train]
        # eval_result = worker_eval.run.remote(6 * 10)
        # eval_result_logger_datum = convert_reward_to_logger_datum.remote(eval_result)
        samples_added = [replay_buffer.add_samples.remote(s) for s in samples]
        while samples_added:
            _, samples_added = ray.wait(samples_added)
            for _ in range(config.num_samples_per_update):
                batch = replay_buffer.get_batch.remote(config.batch_size)
                param_opt.update.remote(batch)

            for w in workers_train:
                w.set_params_and_state.remote(param_opt.get_params_and_state.remote())

        param_opt.save.remote(f"{config.save_dir}/{epoch}.pkl")
        done = param_opt.log.remote()
        jaxboard_logger.write.remote(replay_buffer.get_stats.remote())
        # done = jaxboard_logger.write.remote(eval_result_logger_datum)
        # done = terminal_logger.write.remote(eval_result_logger_datum)

    ray.get(done)

# %%
