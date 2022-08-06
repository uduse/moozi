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
from moozi.core import make_env_and_spec
from moozi.core.scalar_transform import ScalarTransform, make_scalar_transform
from moozi.core.tape import exclude, include, make_tape
from moozi.laws import *
from moozi.laws import MinAtarVisualizer
from moozi.nn.nn import NNModel, make_model
from moozi.nn.training import make_training_suite
from moozi.planner import make_planner
from moozi.universe import Universe
from omegaconf import OmegaConf

load_dotenv()

logger.remove()
logger.add(sys.stderr, level="INFO")


def get_config(overrides={}, path=None):
    if path is None:
        path = Path(__file__).parent / "config.yml"
    config = OmegaConf.load(path)
    if config.dim_action == "auto":
        _, env_spec = make_env_and_spec(config.env.name)
        config.dim_action = env_spec.actions.num_values + 1
    num_rows, num_cols, num_channels = env_spec.observations.shape
    if config.env.num_rows == "auto":
        config.env.num_rows = num_rows
    if config.env.num_cols == "auto":
        config.env.num_cols = num_cols
    if config.env.num_channels == "auto":
        config.env.num_channels = num_channels
    if config.nn.spec_kwargs.obs_channels == "auto":
        config.nn.spec_kwargs.obs_channels = config.num_stacked_frames * (
            config.env.num_channels + config.dim_action
        )
    for key, value in overrides.items():
        OmegaConf.update(config, key, value)
    OmegaConf.resolve(config)
    return config


def get_model(config) -> NNModel:
    scalar_transform = make_scalar_transform(**config.scalar_transform)
    nn_arch_cls = eval(config.nn.arch_cls)
    nn_spec = eval(config.nn.spec_cls)(
        **config.nn.spec_kwargs,
        scalar_transform=scalar_transform,
    )
    return make_model(nn_arch_cls, nn_spec)


def make_env_worker_universe(config, idx: int = 0):
    model = get_model(config)
    num_envs = config.env_worker.num_envs
    vec_env = make_vec_env(config.env.name, num_envs)
    obs_processor = make_obs_processor(
        num_rows=config.env.num_rows,
        num_cols=config.env.num_cols,
        num_channels=config.env.num_channels,
        num_stacked_frames=config.num_stacked_frames,
        dim_action=config.dim_action,
    ).vmap(batch_size=num_envs)
    planner = make_planner(
        model=model,
        batch_size=num_envs,
        **config.env_worker.planner,
    )
    policy = sequential([obs_processor, planner])
    if not config.debug:
        policy = policy.jit(max_trace=1, backend="gpu")
    if idx == 0:
        recorder = make_min_atar_gif_recorder(
            n_channels=config.env.num_channels,
            root_dir="env_worker_gifs",
        )
    else:
        recorder = Law.wrap(lambda: {})

    final_law = sequential(
        [
            vec_env,
            policy,
            make_traj_writer(num_envs),
            recorder,
            make_steps_waiter(config.env_worker.num_steps),
        ]
    )
    tape = make_tape(seed=config.seed + idx)
    tape.update(final_law.malloc())
    return Universe(tape, final_law)


def make_test_worker_universe(config, idx: int = 0):
    model = get_model(config)
    vec_env = make_vec_env(config.env.name, 1)
    obs_processor = make_obs_processor(
        num_rows=config.env.num_rows,
        num_cols=config.env.num_cols,
        num_channels=config.env.num_channels,
        num_stacked_frames=config.num_stacked_frames,
        dim_action=config.dim_action,
    ).vmap(batch_size=1)
    planner = make_planner(model=model, **config.test_worker.planner)
    final_law = sequential(
        [
            vec_env,
            sequential([obs_processor, planner]).jit(backend="gpu"),
            make_min_atar_gif_recorder(
                n_channels=config.env.num_channels,
                root_dir="test_worker_gifs",
            ),
            make_reward_terminator(config.test_worker.num_trajs),
        ]
    )
    tape = make_tape(seed=config.seed + 50 + idx)
    tape.update(final_law.malloc())
    return Universe(tape, final_law)


def make_reanalyze_universe(config, idx: int = 0):
    model = get_model(config)
    batch_size = config.reanalyze.num_envs
    env_mocker = make_batch_env_mocker(batch_size)
    obs_processor = make_obs_processor(
        num_rows=config.env.num_rows,
        num_cols=config.env.num_cols,
        num_channels=config.env.num_channels,
        num_stacked_frames=config.num_stacked_frames,
        dim_action=config.dim_action,
    ).vmap(batch_size)
    planner = make_planner(
        model=model,
        batch_size=batch_size,
        **config.reanalyze.planner,
    )
    policy = sequential(
        [
            obs_processor,
            planner,
            Law.wrap(lambda next_action: {"action": next_action}),
        ]
    )
    if not config.debug:
        policy = policy.jit(max_trace=1, backend="gpu")
    final_law = sequential(
        [
            env_mocker,
            policy,
            make_traj_writer(batch_size),
            make_steps_waiter(config.reanalyze.num_steps),
        ]
    )
    tape = make_tape(seed=config.seed + 100 + idx)
    tape.update(final_law.malloc())
    return Universe(tape, final_law)


def training_suite_factory(config):
    scalar_transform = make_scalar_transform(**config.scalar_transform)
    nn_arch_cls = eval(config.nn.arch_cls)
    nn_spec = eval(config.nn.spec_cls)(
        **config.nn.spec_kwargs,
        scalar_transform=scalar_transform,
    )
    return partial(
        make_training_suite,
        seed=config.seed,
        nn_arch_cls=nn_arch_cls,
        nn_spec=nn_spec,
        weight_decay=config.train.weight_decay,
        lr=config.train.lr,
        num_unroll_steps=config.num_unroll_steps,
        num_stacked_frames=config.num_stacked_frames,
        target_update_period=config.train.target_update_period,
        consistency_loss_coef=config.train.consistency_loss_coef,
    )
