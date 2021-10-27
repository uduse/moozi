from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np

from .types import PolicyFeed


@dataclass
class Tape:
    # meta
    universe_id: int = -1
    num_ticks: int = 0
    num_episodes: int = 0
    avg_episodic_reward: float = 0
    sum_episodic_reward: float = 0

    # environment
    obs: np.ndarray = None
    is_first: bool = True
    is_last: bool = False
    to_play: int = -1
    reward: float = 0.0
    action: int = -1
    discount: float = 1.0
    legal_actions_mask: np.ndarray = None

    # planner
    root_value: float = 0
    action_probs: np.ndarray = None

    # player
    stacked_frames: np.ndarray = None
    policy_feed: Optional[PolicyFeed] = None

    output_buffer: tuple = field(default_factory=tuple)
