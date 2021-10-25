from typing import Any, Callable, Optional
import optax
import ray
from dataclasses import dataclass, field, InitVar


@dataclass
class Config:
    seed: int = 0

    # TODO: replace with str and env getter
    # env_factory: Optional[Callable] = None
    env: str = ''
    # env_spec: Any = None
    # artifact_factory: Optional[Callable] = None

    num_rollout_workers: int = 1
    num_rollout_universes_per_worker: int = 1

    num_stacked_frames: int = 1
    num_unroll_steps: int = 5
    num_td_steps: int = 1
    discount: float = 1.0

    dim_repr: int = 10
    weight_decay: float = 1e-4

    batch_size: int = 256
    lr: float = 2e-3

    replay_buffer_size: int = 1000

    num_epochs: int = 1
    num_ticks_per_epoch: int = 1
    num_updates_per_samples_added: int = 1
