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

load_dotenv()

logger.remove()
logger.add(sys.stderr, level="INFO")

config_path = "/moozi/examples/minatar_space_invaders/config.yml"
config = OmegaConf.load(config_path)
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


# %%
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
            penalizer,
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
            make_reward_terminator(5),
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
        model=model,
        dim_action=config.dim_action,
        batch_size=1,
        num_simulations=10,
        output_action=False,
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
ps = ParameterServer(training_suite_factory=training_suite_factory(config))
rb = ReplayBuffer(**config.replay)
train_worker = RolloutWorker(partial(make_env_worker_universe, config))
test_worker = RolloutWorker(partial(make_test_worker_universe, config))
reanalyze_workers = RolloutWorker(partial(make_reanalyze_universe))

# %%
train_worker.set("params", ps.get_params())
train_worker.set("state", ps.get_state())

# %%
trajs = train_worker.run()
rb.add_trajs(trajs)

# %%
traj = rb.get_trajs_batch(1)[0]

# %%
steps = unstack_sequence_fields(traj, batch_size=traj.frame.shape[0])

# %%
from IPython.display import display
vis = MinAtarVisualizer()
for step in steps:
    img = vis.make_image(frame=step.frame)
    img = vis.add_descriptions(
        img,
        root_value=step.root_value,
        reward=step.last_reward,
        action=step.action,
        action_probs=step.action_probs,
    )
    display(img)

# %%
rb.min_size = 1
target = rb.get_train_targets_batch(10)
print(target.shapes())

# %%
ps.update(target, target.frame.shape[0])

# %%
