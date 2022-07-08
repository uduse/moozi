# %%
import contextlib
import operator
import sys
import uuid
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Set, Tuple, Union

import chex
import gym
import jax
import jax.numpy as jnp
import mctx
import moozi as mz
import numpy as np
import pygraphviz
import ray
import seaborn as sns
import tree
from acme.jax.utils import add_batch_dim
from acme.utils.tree_utils import stack_sequence_fields
from dotenv import load_dotenv
from IPython.display import display
from loguru import logger
from moozi.core import scalar_transform
from moozi.core.env import make_env
from moozi.core.link import link
from moozi.core.scalar_transform import ScalarTransform, make_scalar_transform
from moozi.core.tape import exclude, include, make_tape
from moozi.laws import *
from moozi.laws import MinAtarVisualizer
from moozi.logging import JAXBoardLoggerRemote, TerminalLoggerRemote
from moozi.nn.nn import NNModel, make_model
from moozi.nn.training import (
    _make_obs_from_train_target,
    make_target_from_traj,
    make_training_suite,
)
from moozi.parameter_optimizer import ParameterOptimizer, ParameterServer
from moozi.planner import make_planner
from moozi.replay import ReplayBuffer

# from moozi.rollout_worker import RolloutWorkerWithWeights, make_rollout_workers
from moozi.utils import WallTimer
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont, ImageOps

# %%
load_dotenv()

logger.remove()
logger.add(sys.stderr, level="INFO")

path = "/moozi/examples/minatar_space_invaders/config.yml"
config = OmegaConf.load(path)
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


def make_reanalyze_universe(config):
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
        **config.reanalyze_worker.planner,
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


def make_context_prep(num_stacked_frames, num_unroll_steps, dim_action):
    @Law.wrap
    def prepare_context(train_target):
        obs = _make_obs_from_train_target(
            batch=train_target,
            step=0,
            num_stacked_frames=num_stacked_frames,
            num_unroll_steps=num_unroll_steps,
            dim_action=dim_action,
        )
        return {"obs": obs}

    return prepare_context


@Law.wrap
def output_updated_train_target(
    train_target: TrainTarget, root_value, action_probs, output_buffer
):
    new_target = train_target._replace(root_value=root_value, action_probs=action_probs)
    return {"output_buffer": output_buffer + (new_target,), "quit": True}


# def make_reanalyze_universe_v2(config):
#     # context_prep = make_context_prep(
#     #     num_stacked_frames=config.num_stacked_frames,
#     #     num_unroll_steps=config.num_unroll_steps,
#     #     dim_action=config.dim_action,
#     # )
#     planner = make_planner(
#         model=model,
#         **config.train.reanalyze_worker.planner,
#     ).jit(backend="cpu", max_trace=10)
#     final_law = sequential(
#         [
#             context_prep,
#             planner,
#             output_updated_train_target,
#         ]
#     )
#     tape = make_tape(seed=config.seed)
#     tape.update(final_law.malloc())
#     return Universe(tape, final_law)

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
rb.min_size = 1
vis = MinAtarVisualizer()

# %%
weights_path = "/root/.local/share/virtualenvs/moozi-g1CZ00E9/.guild/runs/a44a467c7d47452691006ebadd50770f/checkpoints/1019.pkl"
ps.restore(weights_path)

# %%
rollout_worker = RolloutWorker(
    partial(make_env_worker_universe, config), name=f"rollout_worker"
)
rollout_worker.set("params", ps.get_params())
rollout_worker.set("state", ps.get_state())

# %%
reanalyze_worker = RolloutWorker(
    partial(make_reanalyze_universe_v2, config), name=f"reanalyze_worker"
)
reanalyze_worker.set("params", ps.get_params())
reanalyze_worker.set("state", ps.get_state())

# %%
trajs = []
for i in range(1):
    trajs.extend(rollout_worker.run())
rb.add_trajs(trajs)

# %%
traj = trajs[0]
targets = [
    make_target_from_traj(
        traj,
        start_idx=i,
        discount=config.discount,
        num_unroll_steps=config.num_unroll_steps,
        num_stacked_frames=config.num_stacked_frames,
        num_td_steps=config.num_td_steps,
    )
    for i in range(traj.action.shape[0])
]

# %%
for i, target in enumerate(targets):
    print(i)
    print(target.n_step_return)
    images = []
    for i in range(target.frame.shape[0]):
        image = vis.make_image(target.frame[i])
        image = vis.add_descriptions(image)
        images.append(image)
    display(vis.cat_images(images))


# %%
images = []
for i in range(traj.frame.shape[0]):
    image = vis.make_image(traj.frame[i])
    image = vis.add_descriptions(
        image,
        action=traj.action[i],
        reward=traj.last_reward[i],
    )
    images.append(image)
vis.cat_images(images)


# %%
images = []
target = targets[18]
for i in range(target.frame.shape[0]):
    image = vis.make_image(target.frame[i])
    image = vis.add_descriptions(image, action=target.action[i])
    images.append(image)
display(vis.cat_images(images))

# %%
obs = _make_obs_from_train_target(
    add_batch_dim(target),
    step=0,
    num_stacked_frames=config.num_stacked_frames,
    num_unroll_steps=config.num_unroll_steps,
    dim_action=config.dim_action,
)
# %%
rb.add_trajs(trajs)

# %%
for _ in range(100):
    print(ps.update(batch, batch_size=config.train.batch_size))

# %%
import random

targets = unstack_sequence_fields(batch, batch.frame.shape[0])
target = random.choice(targets)
target = add_batch_dim(target)

# %%
obs = _make_obs_from_train_target(
    target, 0, config.num_stacked_frames, config.num_unroll_steps, config.dim_action
)

# %%
images = []
for i in range(target.frame.shape[1] - 1):
    image = vis.make_image(target.frame[0, i])
    image = vis.add_descriptions(image, action=target.action[0, i + 1])
    images.append(image)
display(vis.cat_images(images))


# %%
train_target = rb.get_train_targets_batch(10)
reanalyze_worker.set("train_target", train_target)
updated_target = reanalyze_worker.run()

# %%
tree = reanalyze_worker.universe.tape["tree"]

# %%
updated_target = stack_sequence_fields(updated_target)

# %%
from moozi.planner import convert_tree_to_graph

graph = convert_tree_to_graph(tree)

# %%
graph.draw("/tmp/graph.dot", prog="dot")

# %%
updated_target.root_value

# %%
train_target.root_value

# %%
