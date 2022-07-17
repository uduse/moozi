# %%
import sys
import copy
import os
from pathlib import Path
import pprint
import sys
from dataclasses import dataclass, field
from typing import List, Tuple

import moozi
import numpy as np
import ray
from dotenv import load_dotenv
from loguru import logger
from moozi.core import Config, link
from moozi.core.env import make_env
from moozi.laws import (
    FrameStacker,
    MinAtarEnvLaw,
    ReanalyzeEnvLawV2,
    TrajectoryOutputWriter,
    exit_if_no_input,
    increment_tick,
    make_policy_feed,
)
from moozi.logging import JAXBoardLoggerRemote, LogScalar, LogText, TerminalLoggerRemote
from moozi.mcts import ActionSamplerLaw, Planner
from moozi.parameter_optimizer import ParameterOptimizer
from moozi.replay import ReplayBuffer
from moozi.rollout_worker import RolloutWorkerWithWeights, make_rollout_workers
from moozi.utils import WallTimer
from omegaconf import OmegaConf, Container

load_dotenv()

# %%
logger.remove()
logger.add(sys.stderr, level="INFO")

# %%
config = OmegaConf.load(Path(__file__).parent / "config.yml")
OmegaConf.resolve(config)
print(config)

# %%
num_interactions = (
    config.train.num_epochs
    * config.train.num_ticks_per_epoch
    * config.train.num_env_workers
    * config.train.num_universes_per_env_worker
)
logger.info(f"num_interactions: {num_interactions}")

print(config.replay.discount)

# %%


# %%
# class Driver:
#     def __init__(self, config: ):
#         self.config = config
#         self.param_opt = self.make_param_opt()
#         self.replay_buffer = self.make_replay_buffer()
#         # self.train_rollout_workers = self.make_workers()

#     def make_param_opt(self):
#         nn_arch_cls = eval(self.config.nn.arch_cls)
#         nn_spec_cls = eval(self.config.nn.spec_cls)
#         nn_spec = nn_spec_cls(**self.config.nn.spec_kwargs)
#         if self.config.param_opt.use_remote:
#             param_opt = ray.remote(num_gpus=self.config.param_opt.num_gpus)(
#                 ParameterOptimizer
#             ).remote(use_remote=True)
#             param_opt.make_training_suite.remote(
#                 seed=self.config.seed,
#                 nn_arch_cls=nn_arch_cls,
#                 nn_spec=nn_spec,
#                 weight_decay=self.config.train.weight_decay,
#                 lr=self.config.train.lr,
#                 num_unroll_steps=config.num_unroll_steps,
#             )
#             param_opt.make_loggers.remote(
#                 lambda: [
#                     moozi.logging.JAXBoardLoggerV2(name="param_opt"),
#                 ]
#             )
#             return param_opt
#         else:
#             return None

#     def make_replay_buffer(self):
#         if self.config.replay.use_remote:
#             replay_buffer = ray.remote(moozi.replay.ReplayBuffer).remote(
#                 **self.config.replay
#             )
#             return replay_buffer

#     def make_train_rollout_workers(self):
#         workers = []
#         model = self.param_opt.get_model.remote()
#         params_and_state = self.param_opt.get_params_and_state.remote()
#         for i in range(self.config.env_workers.num_workers):
#             worker_name = f"{RolloutWorkerWithWeights.__name__}_{i}"
#             worker = (
#                 ray.remote(RolloutWorkerWithWeights)
#                 .options(
#                     name=worker_name,
#                     num_cpus=self.config.env_workers.num_cpus,
#                     num_gpus=self.config.env_workers.num_gpus,
#                 )
#                 .remote(name=worker_name)
#             )
#             worker.set_model.remote(model)
#             worker.set_params_and_state.remote(params_and_state)
#             # worker.make_batching_layers.remote(num_universes_per_worker)
#             worker.make_universes_from_laws.remote(
#                 laws_factory, num_universes_per_worker
#             )
#             workers.append(worker)
#         return workers

#     def train(self):
#         with WallTimer():
#             for epoch in range(config.train.num_epochs):
#                 for w in workers_env + workers_test + workers_reanalyze:
#                     w.set_params_and_state.remote(
#                         param_opt.get_params_and_state.remote()
#                     )
#                 logger.debug(f"Get params and state scheduled, {len(traj_futures)=}")

#                 while traj_futures:
#                     traj, traj_futures = ray.wait(traj_futures)
#                     traj = traj[0]
#                     replay_buffer.add_trajs.remote(traj)

#                 if epoch >= config.epoch_train_start:
#                     logger.debug(f"Add trajs scheduled, {len(traj_futures)=}")
#                     train_batch = replay_buffer.get_train_targets_batch.remote(
#                         config.big_batch_size
#                     )
#                     logger.debug(
#                         f"Get train targets batch scheduled, {len(traj_futures)=}"
#                     )
#                     update_done = param_opt.update.remote(
#                         train_batch, config.batch_size
#                     )
#                     logger.debug(f"Update scheduled")

#                 jaxboard_logger.write.remote(replay_buffer.get_stats.remote())
#                 env_trajs = [
#                     w.run.remote(config.num_ticks_per_epoch) for w in workers_env
#                 ]
#                 reanalyze_trajs = [w.run.remote(None) for w in workers_reanalyze]
#                 traj_futures = env_trajs + reanalyze_trajs

#                 if enable_test:
#                     if epoch % config.test_interval == 0:
#                         test_result = workers_test[0].run.remote(1000)
#                         test_result_datum = convert_test_result_record.remote(
#                             test_result
#                         )
#                     logger.debug(f"test result scheduled.")

#                 for w in workers_reanalyze:
#                     reanalyze_input = replay_buffer.get_train_targets_batch.remote(
#                         config.num_trajs_per_reanalyze_universe
#                         * config.num_universes_per_reanalyze_worker
#                     )
#                     w.set_inputs.remote(reanalyze_input)
#                 logger.debug(f"reanalyze scheduled.")

#                 if enable_test:
#                     test_done = jaxboard_logger.write.remote(test_result_datum)

#                 param_opt.log.remote()
#                 logger.info(f"Epochs: {epoch + 1} / {config.num_epochs}")

#         if enable_test:
#             logger.debug(ray.get(test_done))
#         ray.get(jaxboard_logger.close.remote())
#         ray.get(param_opt.close.remote())


# config.param_opt.remote = False

# driver = Driver(config)
# driver.make_param_opt()
# driver.run()
