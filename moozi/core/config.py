from ast import Tuple
from dataclasses import asdict, dataclass, field
from os import PathLike
import pprint
from typing import Optional, Type, Union

from moozi.nn import NNArchitecture, NNSpec, ResNetArchitecture, ResNetSpec


@dataclass
class Config:
    seed: int = 0
    env: str = ""

    num_stacked_frames: int = 1
    num_unroll_steps: int = 5
    num_td_steps: int = 1
    discount: float = 1.0

    dirichlet_alpha: float = 0.2
    frac: float = 0.2

    known_bound_min: Optional[float] = None
    known_bound_max: Optional[float] = None

    nn_arch_cls: Type[NNArchitecture] = ResNetArchitecture
    nn_spec: NNSpec = field(init=False)

    # training
    big_batch_size: int = 2048
    batch_size: int = 128
    lr: float = 2e-3
    weight_decay: float = 1e-4

    # replay buffer
    replay_max_size: int = 1_000_000
    replay_min_size: int = 1
    replay_prefetch_max_size: int = 10_000

    # mcts
    num_env_simulations: int = 10
    num_test_simulations: int = 30

    # system configuration
    num_epochs: int = 2
    num_ticks_per_epoch: int = 10
    num_samples_per_update: int = 2

    num_env_workers: int = 2
    num_universes_per_env_worker: int = 2

    num_reanalyze_workers: int = 4
    num_universes_per_reanalyze_worker: int = 2
    num_trajs_per_reanalyze_universe: int = 5

    save_dir: Union[str, PathLike] = "./save/"

    def print(self):
        pprint.pprint(asdict(self))

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        return self
