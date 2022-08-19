from typing import List
from .utils import unstack_sequence_fields_pytree
from .types import StepSample, TrajectorySample


class TrajectoryCollector:
    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.buffer: List[List[StepSample]] = [[] for _ in range(batch_size)]
        self.trajs: List[TrajectorySample] = []

    def add_step_sample(self, step_sample: StepSample) -> "TrajectoryCollector":
        step_sample_flat = unstack_sequence_fields_pytree(step_sample, self.batch_size)
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
