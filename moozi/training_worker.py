from inspect import isclass
import sys
from loguru import logger
from typing import Any, List, Optional, Type, TypeVar, Union

import haiku as hk
import jax
import ray

from moozi.core import TrajectorySample
from moozi.core.history_stacker import HistoryStacker
from moozi.core.trajectory_collector import TrajectoryCollector
from moozi.core.vis import Visualizer, save_gif, next_valid_fpath
from moozi.gii import GII
from moozi.nn import NNModel
from moozi.planner import Planner


class TrainingWorker:
    def __init__(
        self,
        index: int,
        env_name: str,
        num_envs: int,
        model: NNModel,
        stacker: HistoryStacker,
        planner: Planner,
        num_steps: int,
        seed: Optional[int] = None,
        save_gif: bool = True,
        vis: Union[Type[Visualizer], Visualizer, None] = None,
    ):
        self.index = index
        self.seed = seed if seed is not None else index
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.save_gif = save_gif

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
            num_envs=num_envs,
        )
        if isclass(vis):
            _, num_rows, num_cols, _ = self.gii.env.spec.frame.shape
            self.vis = vis(num_rows=num_rows, num_cols=num_cols)
        else:
            self.vis = vis
        self.traj_collector = TrajectoryCollector(num_envs)

        logger.remove()
        logger.add(sys.stderr, level="SUCCESS")
        logger.add(f"logs/training_workers/{index}.debug.log", level="DEBUG")
        logger.add(f"logs/training_workers/{index}.info.log", level="INFO")
        self._flog = logger

    def run(self, epoch: int = 0) -> List[TrajectorySample]:
        samples = [self.gii.tick() for _ in range(self.num_steps)]
        self.traj_collector.add_step_samples(samples)
        trajs = self.traj_collector.flush()
        if self.vis and trajs:
            traj = trajs[0]
            ims = [
                self.vis.make_image(traj.frame[i]) for i in range(traj.frame.shape[0])
            ]
            save_gif(ims, next_valid_fpath("training_workers_gifs", suffix="gif"))
            save_gif(ims, "training_workers_gifs/latest.gif")
        return trajs

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
