from typing import Any, Callable, Optional
import optax
import ray
from dataclasses import dataclass, field, InitVar


@dataclass
class Config:
    env_factory: Optional[Callable] = None
    env_spec: Any = None
    artifact_factory: Optional[Callable] = None

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
    num_updates_per_epoch: int = 1

    # num_ticks: int = 1

    # optimizer_factory: Callable[[], optax.GradientTransformation] = field(init=False)

    # def __post_init__(self, lr: float):
    #     self.optimizer_factory = lambda: optax.adam(lr)


# @dataclass(repr=False)
# class ConfigProxy:
#     handler: None

#     def get(self, name):
#         return ray.get(self.handler.get.remote(name))

#     def set(self, name, val):
#         return ray.get(self.handler.set.remote(name, val))


# _config_global_handler = None


# def get_config_proxy():
#     # TODO: should only be called on driver
#     global _config_global_handler
#     if _config_global_handler is None:
#         _config_global_handler = ray.remote(Config).remote()
#     return ConfigProxy(_config_global_handler)
