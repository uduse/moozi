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
