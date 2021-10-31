from typing import NamedTuple

import numpy as np


class PolicyFeed(NamedTuple):
    features: np.ndarray
    legal_actions_mask: np.ndarray
    random_key: np.ndarray
