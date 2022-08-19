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
        history_length=config.history_length,
        target_update_period=config.train.target_update_period,
        consistency_loss_coef=config.train.consistency_loss_coef,
    )
