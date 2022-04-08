from typing import Deque, List, NamedTuple

import numpy as np
from nptyping import NDArray

# current support:
# - single player games
# - two-player turn-based zero-sum games
class StepSample(NamedTuple):
    frame: NDArray[np.float32]

    # last reward from the environment
    last_reward: NDArray[np.float32]
    is_first: NDArray[np.bool8]
    is_last: NDArray[np.bool8]
    to_play: NDArray[np.int32]
    legal_actions_mask: NDArray[np.int32]

    # root value after the search
    root_value: NDArray[np.float32]
    action_probs: NDArray[np.float32]
    action: NDArray[np.int32]

    def cast(self) -> "StepSample":
        return StepSample(
            frame=np.asarray(self.frame, dtype=np.float32),
            last_reward=np.asarray(self.last_reward, dtype=np.float32),
            is_first=np.asarray(self.is_first, dtype=np.bool8),
            is_last=np.asarray(self.is_last, dtype=np.bool8),
            to_play=np.asarray(self.to_play, dtype=np.int32),
            legal_actions_mask=np.asarray(self.legal_actions_mask, dtype=np.int32),
            root_value=np.asarray(self.root_value, dtype=np.float32),
            action_probs=np.asarray(self.action_probs, dtype=np.float32),
            action=np.asarray(self.action, dtype=np.int32),
        )


# Trajectory is a StepSample with stacked values
class TrajectorySample(StepSample):
    pass
