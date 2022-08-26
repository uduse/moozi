from inspect import isclass
import numpy as np
from pathlib import Path
import random
import sys
from loguru import logger
from typing import Any, List, Optional, Type, TypeVar, Union

import haiku as hk
import jax
import ray

from moozi.core import TrajectorySample
from moozi.core.history_stacker import HistoryStacker
from moozi.core.trajectory_collector import TrajectoryCollector
from moozi.core.vis import Visualizer, next_valid_fpath, visualize_search_tree
from moozi.gii import GII
from moozi.nn import NNModel
from moozi.planner import Planner


class TestingWorker:
    def __init__(
        self,
        index: int,
        env_name: str,
        model: NNModel,
        stacker: HistoryStacker,
        planner: Planner,
        num_steps: int,
        num_envs: int = 1,  # TODO: make this effective
        *,
        seed: Optional[int] = None,
        vis: Union[Type[Visualizer], Visualizer, None] = None,
    ):
        self.index = index
        self.seed = seed if seed is not None else index
        self.num_steps = num_steps

        random_key = jax.random.PRNGKey(self.seed)
        model_key, agent_key = jax.random.split(random_key, 2)
        params, state = model.init_params_and_state(model_key)
        self.gii = GII(
            env_name=env_name,
            stacker=stacker,
            planner=planner,
            params=params,
            state=state,
            random_key=agent_key,
            num_envs=1,
        )
        if isclass(vis):
            _, num_rows, num_cols, _ = self.gii.env.spec.frame.shape
            self.vis = vis(num_rows=num_rows, num_cols=num_cols)
        else:
            self.vis = vis
        self.traj_collector = TrajectoryCollector()

        logger.remove()
        logger.add(sys.stderr, level="SUCCESS")
        logger.add(f"logs/testing_workers/{index}.debug.log", level="DEBUG")
        logger.add(f"logs/testing_workers/{index}.info.log", level="INFO")
        self._flog = logger

    def run(self, epoch: int = 0) -> None:
        self.gii.env_feed = self.gii.env.init()
        for i in range(self.num_steps):
            self.gii.tick()
            search_tree = self.gii.planner_out.tree
            root_state = self.gii.env[0].backend.get_state
            vis_dir = Path(f"search_trees/epoch_{epoch}/step_{i}")
            visualize_search_tree(self.vis, search_tree, root_state, vis_dir)

    def set_params(self, params: hk.Params):
        self.gii.params = params

    def set_state(self, state: hk.State):
        self.gii.state = state

    def set_planner(self, planner: Planner):
        self.gii.planner = planner

    def exec(self, fn):
        return fn(self)

    def get_stats(self) -> dict:
        return {}
