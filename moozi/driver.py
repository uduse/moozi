from dataclasses import dataclass, field
import chex
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import math

import cloudpickle
import haiku as hk
import jax
import ray
from loguru import logger
from omegaconf import DictConfig

import moozi as mz
from moozi.core import HistoryStacker, TrajectoryCollector, make_scalar_transform
from moozi.core.env import (
    GIIEnv,
    GIIEnvFeed,
    GIIEnvOut,
    GIIVecEnv,
    _make_dm_env_and_spec,
)
from moozi.core.scalar_transform import ScalarTransform
from moozi.core.types import TrainingState
from moozi.core.vis import BreakthroughVisualizer, visualize_search_tree
from moozi.nn.nn import NNArchitecture, NNModel, NNSpec
from moozi.parameter_optimizer import ParameterServer
from moozi.planner import Planner
from moozi.replay import ReplayBuffer, ShardedReplayBuffer
from moozi.testing_worker import TestingWorker
from moozi.tournament import Player, Tournament
from moozi.training_worker import TrainingWorker
from omegaconf import OmegaConf


# TODO: separate config parsing process
def get_config(path="config.yml", overrides={}):
    path = Path(path)
    config = OmegaConf.load(path)
    env = GIIEnv.new(config.env.name)

    num_rows, num_cols, num_channels = env.spec.frame.shape
    if config.dim_action == "auto":
        config.dim_action = env.spec.dim_action
    if config.env.num_rows == "auto":
        config.env.num_rows = num_rows
    if config.env.num_cols == "auto":
        config.env.num_cols = num_cols
    if config.env.num_channels == "auto":
        config.env.num_channels = num_channels
    if config.env.num_players == "auto":
        config.env.num_players = env.spec.num_players

    config.num_steps_per_epoch = (
        config.training_worker.num_steps
        * config.training_worker.num_workers
        * config.training_worker.num_envs
    ) + (
        config.reanalyze_worker.num_workers
        * config.reanalyze_worker.num_envs
        * config.reanalyze_worker.num_steps
    )
    config.num_env_steps_per_epoch = (
        config.training_worker.num_steps
        * config.training_worker.num_workers
        * config.training_worker.num_envs
    )
    config.num_updates = int(
        config.train.update_step_ratio
        * config.num_steps_per_epoch
        / config.train.batch_size
    )

    for key, value in overrides.items():
        OmegaConf.update(config, key, value)
    OmegaConf.resolve(config)
    return config


@dataclass
class Driver:
    # static properties
    config: DictConfig
    model: mz.nn.NNModel
    stacker: HistoryStacker

    training_planner: Planner
    testing_planner: Planner

    # dynamic properties
    # trianing workers
    training_workers: Optional[List[TrainingWorker]] = None
    testing_workers: Optional[List[TrainingWorker]] = None

    ps: Optional[ParameterServer] = None
    rb: Union[ReplayBuffer, ShardedReplayBuffer, None] = None

    training_start: bool = False
    trajs: list = field(default_factory=list)
    epoch: int = 0

    @staticmethod
    def setup(config: DictConfig):
        scalar_transform = make_scalar_transform(**config.scalar_transform)
        nn_arch_cls: Type[NNArchitecture] = mz.nn.get(config.nn.arch_cls)
        nn_spec: NNSpec = mz.nn.get(config.nn.spec_cls)(
            **config.nn.spec_kwargs,
            scalar_transform=scalar_transform,
        )
        model = mz.nn.make_model(nn_arch_cls, nn_spec)

        stacker = HistoryStacker(
            num_rows=config.env.num_rows,
            num_cols=config.env.num_cols,
            num_channels=config.env.num_channels,
            history_length=config.history_length,
            dim_action=config.dim_action,
        )
        training_planner = Planner(
            batch_size=config.training_worker.num_envs,
            model=model,
            num_players=config.env.num_players,
            **config.training_worker.planner,
        )
        testing_planner = Planner(
            batch_size=1,
            model=model,
            num_players=config.env.num_players,
            **config.testing_worker.planner,
        )
        config = config.copy()
        return Driver(
            config=config,
            model=model,
            stacker=stacker,
            training_planner=training_planner,
            testing_planner=testing_planner,
        )

    def start(self):
        self._start_replay_buffer()
        self._start_parameter_server()
        self._start_workers()

    def _start_workers(self):
        self.training_workers = [
            ray.remote(
                num_gpus=self.config.training_worker.num_gpus,
                num_cpus=self.config.training_worker.num_cpus,
            )(TrainingWorker).remote(
                index=i,
                seed=i,
                env_name=self.config.env.name,
                num_envs=self.config.training_worker.num_envs,
                model=self.model,
                stacker=self.stacker,
                planner=self.training_planner,
                num_steps=self.config.training_worker.num_steps,
                use_vis=(i == 0),
            )
            for i in range(self.config.training_worker.num_workers)
        ]
        self.testing_workers = [
            ray.remote(
                num_gpus=self.config.testing_worker.num_gpus,
                num_cpus=self.config.testing_worker.num_cpus,
            )(TestingWorker).remote(
                index=0,
                env_name=self.config.env.name,
                model=self.model,
                stacker=self.stacker,
                planner=self.testing_planner,
                num_steps=self.config.testing_worker.num_steps,
                use_vis=True
            )
        ]

    def _start_parameter_server(self):
        self.ps = ray.remote(num_gpus=self.config.param_opt.num_gpus)(
            ParameterServer
        ).remote(
            partial(
                mz.nn.training.make_training_suite,
                seed=self.config.seed,
                model=self.model,
                weight_decay=self.config.train.weight_decay,
                lr=self.config.train.lr,
                num_unroll_steps=self.config.num_unroll_steps,
                history_length=self.config.history_length,
                target_update_period=self.config.train.target_update_period,
                consistency_loss_coef=self.config.train.consistency_loss_coef,
            ),
            load_from=self.config.param_opt.load_from,
        )

    def _start_replay_buffer(self):
        if self.config.replay.num_shards > 1:
            self.rb = ray.remote(ShardedReplayBuffer).remote(
                num_shards=self.config.replay.num_shards,
                kwargs=self.config.replay.kwargs,
            )
        else:
            self.rb = ray.remote(ReplayBuffer).remote(**self.config.replay.kwargs)

    def wait(self):
        ray.get(
            [w.get_stats.remote() for w in self.training_workers + self.testing_workers]
            + [self.ps.get_stats.remote(), self.rb.get_stats.remote()]
        )

    def sync_params_and_state(self):
        if self.epoch == 0:
            for w in self.training_workers + self.testing_workers:
                w.set_params.remote(self.ps.get_params.remote())
                w.set_state.remote(self.ps.get_state.remote())
        else:
            if self.epoch % self.config.training_worker.update_period == 0:
                for w in self.training_workers:
                    w.set_params.remote(self.ps.get_params.remote())
                    w.set_state.remote(self.ps.get_state.remote())
            if self.epoch % self.config.testing_worker.update_period == 0:
                for w in self.testing_workers:
                    w.set_params.remote(self.ps.get_params.remote())
                    w.set_state.remote(self.ps.get_state.remote())

    def generate_trajs(self):
        self.trajs.clear()
        for w in self.training_workers:
            self.trajs.append(w.run.remote(self.epoch))

    def update_parameters(self):
        if self.training_start:
            samples_per_update = self.config.train.batch_size * self.config.num_updates
            # TODO: implement sharded replay buffer sampling
            # if self.config.replay.num_shards > 1:
            #     for batch in self.rb.get_train_targets_batch_sharded.remote():
            #         self.ps.update.remote(batch, batch_size=self.config.train.batch_size)
            # else:
            batch = self.rb.get_train_targets_batch.remote(
                batch_size=samples_per_update
            )
            self.ps.update.remote(batch, batch_size=self.config.train.batch_size)
        else:
            if self.config.replay.num_shards > 1:
                stats = ray.get(self.rb.get_stats.remote())
                num_targets = sum([s["targets_size"] for s in stats.values()])
            else:
                num_targets = ray.get(self.rb.get_stats.remote())["targets_size"]
            if num_targets >= self.config.train.min_targets_to_train:
                self.training_start = True
                logger.info("Training start")

    def process_trajs(self):
        for trajs in self.trajs:
            self.rb.add_trajs.remote(trajs, from_env=True)

    def house_keep(self):
        self.ps.log_tensorboard.remote()
        self.rb.log_tensorboard.remote()
        if self.epoch % self.config.param_opt.save_interval == 0:
            self.ps.save.remote()

    def run_tests(self):
        if self.epoch % self.config.testing_worker.test_period == 0:
            for w in self.testing_workers:
                w.run.remote(self.epoch)

    def schedule_one_epoch(self):
        self.sync_params_and_state()
        self.update_parameters()
        self.process_trajs()
        self.generate_trajs()
        self.run_tests()
        self.house_keep()
        self.epoch += 1

    def run(self):
        for i in range(self.config.train.num_epochs):
            if i % 20 == 0:
                self.wait()
                logger.info(f"Epoch {i} done.")
            self.schedule_one_epoch()
        self.wait()


@dataclass
class ConfigFactory:
    config: DictConfig

    def make_env(self) -> GIIEnv:
        return GIIEnv.new(self.config.env.name)

    def make_scalar_transform(self) -> ScalarTransform:
        return make_scalar_transform(**self.config.scalar_transform)

    def make_model(self) -> NNModel:
        nn_arch_cls: Type[NNArchitecture] = mz.nn.get(self.config.nn.arch_cls)
        nn_spec: NNSpec = mz.nn.get(self.config.nn.spec_cls)(
            **self.config.nn.spec_kwargs,
            scalar_transform=self.make_scalar_transform(),
        )
        return mz.nn.make_model(nn_arch_cls, nn_spec)

    def make_history_stacker(self) -> HistoryStacker:
        return HistoryStacker(
            num_rows=self.config.env.num_rows,
            num_cols=self.config.env.num_cols,
            num_channels=self.config.env.num_channels,
            history_length=self.config.history_length,
            dim_action=self.config.dim_action,
        )

    def make_training_planner(self) -> Planner:
        return Planner(
            batch_size=self.config.training_worker.num_envs,
            model=self.make_model(),
            num_players=self.config.env.num_players,
            **self.config.training_worker.planner,
        )

    def make_testing_planner(self) -> Planner:
        return Planner(
            batch_size=1,
            model=self.make_model(),
            num_players=self.config.env.num_players,
            **self.config.testing_worker.planner,
        )

    def make_random_key(self) -> chex.PRNGKey:
        return jax.random.PRNGKey(self.config.seed)