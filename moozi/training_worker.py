from typing import Any, List, Optional, Type, TypeVar, Union

import haiku as hk
import jax
import ray

from moozi.core import TrajectorySample
from moozi.core.history_stacker import HistoryStacker
from moozi.core.trajectory_collector import TrajectoryCollector
from moozi.core.vis import Visualizer
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
        seed: Optional[int] = 0,
        save_gif: bool = True,
        vis_cls: Optional[Type[Visualizer]] = None,
    ):
        if seed is None:
            seed = index
        self.index = index
        self.seed = seed
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
        if vis_cls:
            self.vis = vis_cls(5, 6)
        self.traj_collector = TrajectoryCollector(num_envs)

    def run(self) -> List[TrajectorySample]:
        samples = [self.gii.tick() for _ in range(self.num_steps)]
        self.traj_collector.add_step_samples(samples)
        return self.traj_collector.flush()

    def set_params(self, params: Union[hk.Params, ray.ObjectRef]):
        if isinstance(params, ray.ObjectRef):
            params = ray.get(params)
        self.gii.params = params        
    
    def set_state(self, state: Union[hk.Params, ray.ObjectRef])
        pass
