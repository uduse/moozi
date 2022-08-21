# %%
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Type

import cloudpickle
import haiku as hk
import jax
import moozi as mz
import ray
from loguru import logger
from moozi.core import HistoryStacker, TrajectoryCollector, make_scalar_transform
from moozi.core.env import GIIEnv, GIIEnvFeed, GIIEnvOut, GIIVecEnv
from moozi.core.types import TrainingState
from moozi.core.vis import BreakthroughVisualizer, visualize_search_tree
from moozi.nn.nn import NNArchitecture, NNSpec
from moozi.parameter_optimizer import ParameterServer
from moozi.planner import Planner
from moozi.replay import ReplayBuffer, ShardedReplayBuffer
from moozi.tournament import Candidate, Tournament
from moozi.training_worker import TrainingWorker
from omegaconf import DictConfig

from lib import get_config

# %%
config = get_config()


@dataclass
class Driver:
    config: DictConfig
    model: mz.nn.NNModel
    stacker: HistoryStacker
    trp: Planner
    # tsp: Planner

    trws: Optional[List[TrainingWorker]] = None
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
        trp = Planner(
            batch_size=config.training_worker.num_envs,
            dim_action=config.dim_action,
            model=model,
            discount=config.discount,
            num_unroll_steps=config.num_unroll_steps,
            num_simulations=config.training_worker.planner.num_simulations,
            limit_depth=True,
        )
        config = config.copy()
        return Driver(
            config=config,
            model=model,
            stacker=stacker,
            trp=trp,
        )

    def start(self):
        if config.replay.num_shards > 1:
            self.rb = ray.remote(ShardedReplayBuffer).remote(
                num_shards=config.replay.num_shards, kwargs=config.replay.kwargs
            )
        else:
            self.rb = ray.remote(ReplayBuffer).remote(**config.replay.kwargs)
        self.ps = ray.remote(num_gpus=config.param_opt.num_gpus)(
            ParameterServer
        ).remote(
            partial(
                mz.nn.training.make_training_suite,
                seed=config.seed,
                model=self.model,
                weight_decay=config.train.weight_decay,
                lr=config.train.lr,
                num_unroll_steps=config.num_unroll_steps,
                history_length=config.history_length,
                target_update_period=config.train.target_update_period,
                consistency_loss_coef=config.train.consistency_loss_coef,
            )
        )
        self.trws = [
            ray.remote(num_gpus=config.training_worker.num_gpus)(TrainingWorker).remote(
                index=i,
                env_name=config.env.name,
                num_envs=config.training_worker.num_envs,
                model=self.model,
                stacker=self.stacker,
                planner=self.trp,
                num_steps=config.training_worker.num_steps,
                vis=BreakthroughVisualizer if i == 0 else None,
            )
            for i in range(config.training_worker.num_workers)
        ]

    def wait(self):
        ray.get(
            [w.get_stats.remote() for w in self.trws]
            + [self.ps.get_stats.remote(), self.rb.get_stats.remote()]
        )

    def sync_params_and_state(self):
        if self.epoch == 0:
            for w in self.trws:
                w.set_params.remote(self.ps.get_params.remote())
                w.set_state.remote(self.ps.get_state.remote())
        else:
            if self.epoch % self.config.training_worker.update_period == 0:
                for w in self.trws:
                    w.set_params.remote(self.ps.get_params.remote())
                    w.set_state.remote(self.ps.get_state.remote())

    def generate_trajs(self):
        self.trajs.clear()
        for w in self.trws:
            self.trajs.append(w.run.remote())

    def update_parameters(self):
        if self.training_start:
            batch = self.rb.get_train_targets_batch.remote(
                batch_size=config.train.batch_size * self.config.num_updates
            )
            self.ps.update.remote(batch, batch_size=config.train.batch_size)
        else:
            if config.replay.num_shards > 1:
                stats = ray.get(self.rb.get_stats.remote())
                num_targets = sum([s['targets_size'] for s in stats.values()])
            else:
                num_targets = ray.get(self.rb.get_stats.remote())['targets_size']
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
        


# %%
driver = Driver.setup(config)
driver.start()

# %%
for i in range(100):
    driver.run_one_epoch()

# %%
