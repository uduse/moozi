from typing import Deque, List, NamedTuple

import numpy as np
from nptyping import NDArray


class TrainTarget(NamedTuple):
    # right now we only support perfect information games
    # so stacked_frames is a history of symmetric observations
    stacked_frames: NDArray[np.float32]

    # action taken in in each step, -1 means no action taken (terminal state)
    action: NDArray[np.int32]

    # value is computed based on the player of each timestep instead of the
    # player at the first timestep as the root player
    # this means if all rewards are positive, the values are always positive too
    value: NDArray[np.float32]

    # a faithful slice of the trajectory rewards, not flipped for multi-player games
    last_reward: NDArray[np.float32]

    # action probabilities from the search result
    action_probs: NDArray[np.float32]

    def cast(self) -> "TrainTarget":
        return TrainTarget(
            stacked_frames=np.asarray(self.stacked_frames, dtype=np.float32),
            action=np.asarray(self.action, dtype=np.int32),
            value=np.asarray(self.value, dtype=np.float32),
            last_reward=np.asarray(self.last_reward, dtype=np.float32),
            action_probs=np.asarray(self.action_probs, dtype=np.float32),
        )
