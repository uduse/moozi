# %%
import contextlib
import operator
import sys
import uuid
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable, List, Set, Tuple, Union

import moozi
from acme.utils.tree_utils import stack_sequence_fields
from dotenv import load_dotenv
from loguru import logger
from moozi.core import scalar_transform
from moozi.core.scalar_transform import ScalarTransform, make_scalar_transform
from moozi.core.tape import exclude, include, make_tape
from moozi.laws import *
from moozi.laws import MinAtarVisualizer
from moozi.nn.nn import NNModel, make_model
from moozi.nn.training import make_training_suite
from moozi.planner import make_planner, make_gumbel_planner
from moozi.universe import Universe
from omegaconf import OmegaConf

load_dotenv()

logger.remove()
logger.add(sys.stderr, level="INFO")


def get_config():
    config = OmegaConf.load(Path(__file__).parent / "config.yml")
    OmegaConf.resolve(config)
    return config


config = get_config()

scalar_transform = make_scalar_transform(**config.scalar_transform)
nn_arch_cls = eval(config.nn.arch_cls)
nn_spec = eval(config.nn.spec_cls)(
    **config.nn.spec_kwargs, scalar_transform=scalar_transform
)
model = make_model(nn_arch_cls, nn_spec)


def make_env_worker_universe(config):
    num_envs = config.train.env_worker.num_envs
    vec_env = make_vec_env(config.env.name, num_envs)
    frame_stacker = make_batch_stacker_v2(
        num_envs,
        config.env.num_rows,
        config.env.num_cols,
        config.env.num_channels,
        config.num_stacked_frames,
        config.dim_action,
    )

    if config.train.env_worker.planner_type == "gumbel":
        planner = make_gumbel_planner(
            model=model, **config.train.env_worker.planner
        ).jit(backend="gpu", max_trace=10)
    elif config.train.env_worker.planner_type == "muzero":
        planner = make_planner(
            model=model, **config.train.env_worker.planner
        )

    traj_writer = make_traj_writer(num_envs)
    terminator = make_terminator(num_envs)

    final_law = sequential(
        [
            vec_env,
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
    stacker = make_batch_stacker_v2(
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
    planner = make_planner(model=model, **config.train.reanalyze_worker.planner).jit(
        backend="cpu", max_trace=10
    )
    terminalor = make_terminator(size=1)
    final_law = sequential(
        [
            env_mocker,
            stacker,
            concat_stacked_to_obs,
            planner,
            Law.wrap(lambda next_action: {"action": next_action}),
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
