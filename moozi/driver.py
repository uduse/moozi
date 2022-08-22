from dataclasses import dataclass, field
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
from moozi.core.env import GIIEnv, GIIEnvFeed, GIIEnvOut, GIIVecEnv, make_env_and_spec
from moozi.core.types import TrainingState
from moozi.core.vis import BreakthroughVisualizer, visualize_search_tree
from moozi.nn.nn import NNArchitecture, NNSpec
from moozi.parameter_optimizer import ParameterServer
from moozi.planner import Planner
from moozi.replay import ReplayBuffer, ShardedReplayBuffer
from moozi.tournament import Player, Tournament
from moozi.training_worker import TrainingWorker
from omegaconf import OmegaConf


# TODO: separate config parsing process
def get_config(overrides={}, path="config.yml"):
    path = Path(path)
    config = OmegaConf.load(path)
    if config.dim_action == "auto":
        _, env_spec = make_env_and_spec(config.env.name)
        config.dim_action = env_spec.actions.num_values + 1
    try:
        num_rows, num_cols, num_channels = env_spec.observations.shape
    except:
        num_rows, num_cols, num_channels = env_spec.observations.observation.shape
    if config.env.num_rows == "auto":
        config.env.num_rows = num_rows
    if config.env.num_cols == "auto":
        config.env.num_cols = num_cols
    if config.env.num_channels == "auto":
        config.env.num_channels = num_channels

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
    traning_workers: Optional[List[TrainingWorker]] = None
    # trws: Optional[List[TrainingWorker]] = None
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
            **config.training_worker.planner,
        )
        testing_planner = Planner(
            batch_size=1,
            model=model,
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
        self.traning_workers = [
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
                vis=BreakthroughVisualizer if i == 0 else None,
            )
            for i in range(self.config.training_worker.num_workers)
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
            )
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
            [w.get_stats.remote() for w in self.traning_workers]
            + [self.ps.get_stats.remote(), self.rb.get_stats.remote()]
        )

    def sync_params_and_state(self):
        if self.epoch == 0:
            for w in self.traning_workers:
                w.set_params.remote(self.ps.get_params.remote())
                w.set_state.remote(self.ps.get_state.remote())
        else:
            if self.epoch % self.config.training_worker.update_period == 0:
                for w in self.traning_workers:
                    w.set_params.remote(self.ps.get_params.remote())
                    w.set_state.remote(self.ps.get_state.remote())

    def generate_trajs(self):
        self.trajs.clear()
        for w in self.traning_workers:
            self.trajs.append(w.run.remote())

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

    def run_one_epoch(self):
        logger.info(f"epoch: {self.epoch}")
        self.sync_params_and_state()
        self.update_parameters()
        self.process_trajs()
        self.generate_trajs()
        self.house_keep()
        self.wait()
        self.epoch += 1

    def run(self):
        for i in range(self.config.train.num_epochs):
            self.run_one_epoch()
