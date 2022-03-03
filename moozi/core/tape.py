from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import numpy as np

from moozi.nn.nn import NNModel

from .types import PolicyFeed


@dataclass
class Tape:
    universe_id: int = -1

    # statistics
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
    legal_actions_mask: np.ndarray = np.array(1)

    # planner input
    root_inf_fn: Callable = None
    trans_inf_fn: Callable = None
    num_simulations: int = 10

    # planner output
    root_value: float = 0
    action_probs: np.ndarray = np.array(0.0)
    mcts_root: Optional[Any] = None

    # player inputs
    stacked_frames: np.ndarray = np.array(0)
    policy_feed: Optional[PolicyFeed] = None

    output_buffer: tuple = field(default_factory=tuple)
