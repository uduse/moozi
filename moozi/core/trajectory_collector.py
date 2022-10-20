from typing import List
import jax
from .utils import unstack_sequence_fields_pytree
from .types import StepSample, TrajectorySample
from moozi.utils import WallTimer


class TrajectoryCollector:
    def __init__(self, batch_size: int = 1):
        self.batch_size = batch_size
        self.buffer: List[List[StepSample]] = [[] for _ in range(batch_size)]
        self.trajs: List[TrajectorySample] = []
        self._unstacker = jax.jit(unstack_sequence_fields_pytree, backend='cpu', static_argnames=['batch_size'])

    def add_step_sample(self, step_sample: StepSample) -> "TrajectoryCollector":
        step_sample_flat = self._unstacker(step_sample, self.batch_size)
        for new_sample, step_samples in zip(step_sample_flat, self.buffer):
            step_samples.append(new_sample)
            if new_sample.is_last:
                traj = TrajectorySample.from_step_samples(step_samples)
                self.trajs.append(traj)
                step_samples.clear()
        return self

    def add_step_samples(self, step_samples: List[StepSample]) -> "TrajectoryCollector":
        for s in step_samples:
            self.add_step_sample(s)
        return self

    def flush(self) -> List[TrajectorySample]:
        ret = self.trajs.copy()
        self.trajs.clear()
        return ret

    def __len__(self):
        return len(self.trajs)