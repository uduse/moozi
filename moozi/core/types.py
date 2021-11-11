from typing import NamedTuple

import numpy as np


class PolicyFeed(NamedTuple):
    stacked_frames: np.ndarray
    to_play: int
    legal_actions_mask: np.ndarray
    random_key: np.ndarray


BASE_PLAYER: int = 0
