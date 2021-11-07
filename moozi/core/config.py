from dataclasses import asdict, dataclass
import pprint


@dataclass
class Config:
    seed: int = 0

    env: str = ""

    num_stacked_frames: int = 1
    num_unroll_steps: int = 5
    num_td_steps: int = 1
    discount: float = 1.0

    dim_repr: int = 10

    weight_decay: float = 1e-4

    batch_size: int = 256
    lr: float = 2e-3

    replay_buffer_size: int = 1000

    num_rollout_workers: int = 2
    num_rollout_universes_per_worker: int = 2
    num_epochs: int = 2
    num_ticks_per_epoch: int = 10
    num_updates_per_samples_added: int = 2

    def print(self):
        pprint.pprint(asdict(self))

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        return self