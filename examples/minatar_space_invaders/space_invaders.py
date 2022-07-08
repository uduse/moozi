# %%
from moozi.laws import MinAtarVisualizer
import gym
import operator
import uuid
from PIL import Image, ImageDraw, ImageFont
import seaborn as sns
import tree
import contextlib
from functools import partial
import chex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Set, Tuple, Union

import jax

import moozi
import numpy as np
import ray
from acme.utils.tree_utils import stack_sequence_fields
from dotenv import load_dotenv
from loguru import logger
import moozi as mz
from moozi.core import scalar_transform
from moozi.core.env import make_env
from moozi.core.scalar_transform import ScalarTransform, make_scalar_transform
from moozi.core.tape import make_tape, include, exclude
from moozi.logging import JAXBoardLoggerRemote, TerminalLoggerRemote
from moozi.nn.nn import NNModel, make_model
from moozi.nn.training import make_training_suite
from moozi.parameter_optimizer import ParameterOptimizer, ParameterServer
from moozi.replay import ReplayBuffer
from moozi.core.link import link
from moozi.laws import *
from moozi.planner import make_planner

# from moozi.rollout_worker import RolloutWorkerWithWeights, make_rollout_workers
from moozi.utils import WallTimer
from omegaconf import OmegaConf

# %%
load_dotenv()

logger.remove()
logger.add(sys.stderr, level="INFO")

config = OmegaConf.load(Path(__file__).parent / "config.yml")
OmegaConf.resolve(config)
print(OmegaConf.to_yaml(config, resolve=True))


class Universe:
    def __init__(self, tape, law: Law) -> None:
        assert isinstance(tape, dict)
        self.tape = tape
        self.law = law

    def tick(self):
        self.tape = self.law.apply(self.tape)

    def run(self):
        while True:
            self.tick()
            if self.tape["quit"]:
                break
        return self.flush()

    def flush(self):
        ret = self.tape["output_buffer"]
        logger.debug(f"flushing {len(ret)} trajectories")
        self.tape["output_buffer"] = tuple()
        return ret


class RolloutWorker:
    def __init__(
        self, universe_factory: Callable, name: str = "rollout_worker"
    ) -> None:
        self.universe = universe_factory()
        self.name = name
        print(f"{self.name} created")

        from loguru import logger

        logger.remove()
        logger.add(f"logs/rollout_worker.{self.name}.debug.log", level="DEBUG")
        logger.add(f"logs/rollout_worker.{self.name}.info.log", level="INFO")
        logger.info(
            f"RolloutWorker created, name: {self.name}, universe include {self.universe.tape.keys()}"
        )

    def run(self):
        return self.universe.run()

    def set(self, key, value):
        if isinstance(value, ray.ObjectRef):
            value = ray.get(value)
        self.universe.tape[key] = value


scalar_transform = make_scalar_transform(**config.scalar_transform)
nn_arch_cls = eval(config.nn.arch_cls)
nn_spec = eval(config.nn.spec_cls)(
    **config.nn.spec_kwargs, scalar_transform=scalar_transform
)
model = make_model(nn_arch_cls, nn_spec)


def make_env_worker_universe(config):
    num_envs = config.train.env_worker.num_envs
    vec_env = make_vec_env(config.env.name, num_envs)
    frame_stacker = make_batch_stacker(
        num_envs,
        config.env.num_rows,
        config.env.num_cols,
        config.env.num_channels,
        config.num_stacked_frames,
        config.dim_action,
    )

    penalizer = make_last_step_penalizer(penalty=10.0).vmap(batch_size=num_envs)
    planner = make_planner(model=model, **config.train.env_worker.planner).jit(
        backend="gpu", max_trace=10
    )

    traj_writer = make_traj_writer(num_envs)
    terminator = make_terminator(num_envs)

    final_law = sequential(
        [
            vec_env,
            # penalizer,
            frame_stacker,
            concat_stacked_to_obs,
            planner,
            make_min_atar_gif_recorder(n_channels=6, root_dir="env_worker_gifs"),
            traj_writer,
            terminator,
        ]
    )
    tape = make_tape(seed=config.seed)
    tape.update(final_law.malloc())
    return Universe(tape, final_law)


def make_test_worker_universe(config):
    stacker = make_batch_stacker(
        1,
        config.env.num_rows,
        config.env.num_cols,
        config.env.num_channels,
        config.num_stacked_frames,
        config.dim_action,
    )
    planner = make_planner(model=model, **config.train.test_worker.planner).jit(
        backend="gpu", max_trace=10
    )

    final_law = sequential(
        [
            make_vec_env(config.env.name, 1),
            stacker,
            concat_stacked_to_obs,
            planner,
            make_min_atar_gif_recorder(n_channels=6, root_dir="test_worker_gifs"),
            make_traj_writer(1),
            make_reward_terminator(1),
        ]
    )
    tape = make_tape(seed=config.seed)
    tape.update(final_law.malloc())
    return Universe(tape, final_law)


def make_reanalyze_universe():
    env_mocker = make_env_mocker()
    stacker = make_stacker(
        config.env.num_rows,
        config.env.num_cols,
        config.env.num_channels,
        config.num_stacked_frames,
        config.dim_action,
    ).vmap(batch_size=1)
    planner = make_planner(
        model=model, **config.train.reanalyze_worker.planner
    ).jit(backend="cpu", max_trace=10)
    terminalor = make_terminator(size=1)
    final_law = sequential(
        [
            env_mocker,
            stacker,
            concat_stacked_to_obs,
            planner,
            make_traj_writer(1),
            terminalor,
        ]
    )
    tape = make_tape(seed=config.seed)
    tape.update(final_law.malloc())
    return Universe(tape, final_law)


def training_suite_factory(config):
    return partial(
        make_training_suite,
        seed=config.seed,
        nn_arch_cls=nn_arch_cls,
        nn_spec=nn_spec,
        weight_decay=config.train.weight_decay,
        lr=config.train.lr,
        num_unroll_steps=config.num_unroll_steps,
        num_stacked_frames=config.num_stacked_frames,
    )


# %%
ps = ray.remote(num_gpus=config.param_opt.num_gpus)(ParameterServer).remote(
    training_suite_factory=training_suite_factory(config), use_remote=True
)
rb = ray.remote(ReplayBuffer).remote(**config.replay)

# %%
train_workers = [
    ray.remote(num_gpus=config.train.env_worker.num_gpus)(RolloutWorker).remote(
        partial(make_env_worker_universe, config), name=f"rollout_worker_{i}"
    )
    for i in range(config.train.env_worker.num_workers)
]

# %%
test_worker = ray.remote(num_gpus=config.train.test_worker.num_gpus)(
    RolloutWorker
).remote(partial(make_test_worker_universe, config), name="test_worker")

# %%
reanalyze_workers = [
    ray.remote(num_gpus=0, num_cpus=0)(RolloutWorker).remote(
        partial(make_reanalyze_universe), name=f"reanalyze_worker_{i}"
    )
    for i in range(config.train.reanalyze_worker.num_workers)
]


@ray.remote(
    num_gpus=0, num_cpus=0, num_returns=config.train.reanalyze_worker.num_workers
)
def dispatch_trajs(trajs: list):
    return trajs


# %%
jb_logger = JAXBoardLoggerRemote.remote()
terminal_logger = TerminalLoggerRemote.remote()
start_training = False
train_targets = []
for epoch in range(1, config.train.num_epochs + 1):
    logger.info(f"Epoch {epoch}")

    for w in train_workers + reanalyze_workers:
        w.set.remote("params", ps.get_params.remote())
        w.set.remote("state", ps.get_state.remote())

    if epoch % config.train.test_worker.interval == 0:
        # launch test
        test_worker.set.remote("params", ps.get_params.remote())
        test_worker.set.remote("state", ps.get_state.remote())
        test_result = test_worker.run.remote()
        terminal_logger.write.remote(test_result)
        jb_logger.write.remote(test_result)

    # sync
    # ray.get(train_targets)

    # generate train targets
    train_targets.clear()
    for w in train_workers:
        sample = w.run.remote()
        train_targets.append(rb.add_trajs.remote(sample, from_env=True))

    if not start_training:
        rb_size = ray.get(rb.get_num_targets_created.remote())
        start_training = rb_size >= config.replay.min_size
        if start_training:
            logger.info(f"Start training ...")

    if start_training:
        desired_num_updates = (
            config.train.sample_update_ratio
            * ray.get(rb.get_num_targets_created.remote())
            / config.train.batch_size
        )
        num_updates = int(desired_num_updates - ray.get(ps.get_training_steps.remote()))
        batch = rb.get_train_targets_batch.remote(
            batch_size=config.train.batch_size * num_updates
        )
        ps_update_result = ps.update.remote(batch, batch_size=config.train.batch_size)
        terminal_logger.write.remote(ps_update_result)

        if config.train.reanalyze_worker.num_workers > 0:
            traj_refs = dispatch_trajs.remote(
                rb.get_trajs_batch.remote(config.train.reanalyze_worker.num_workers)
            )
            for i, re_w in enumerate(reanalyze_workers):
                re_w.set.remote("traj", traj_refs[0])
                updated_traj = re_w.run.remote()
                rb.add_trajs.remote(updated_traj, from_env=False)

    if epoch % config.param_opt.save_interval == 0:
        ps.save.remote()

    ray.timeline(filename="/tmp/timeline.json")
    ps.log_tensorboard.remote()
    jb_logger.write.remote(rb.get_stats.remote())

# %%
