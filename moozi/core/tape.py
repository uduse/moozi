import jax
import jax.numpy as jnp
from typing import Any, Callable, Dict, Optional
import numpy as np

from moozi.nn.nn import NNModel

from .types import PolicyFeed


def make_tape(num_envs: int, dim_actions: int, num_stacked_frames: int, seed: int = 0):
    tape = {}
    tape["root_value"] = jnp.zeros(num_envs, dtype=jnp.float32)
    tape["random_key"] = jax.random.PRNGKey(seed)
    tape["output_buffer"] = tuple()
    return tape


# # TODO: fix type hints here
# @dataclass
# class Tape:
#     # statistics
#     num_ticks: int = 0
#     num_episodes: int = 0
#     avg_episodic_reward: float = 0
#     sum_episodic_reward: float = 0

#     # environment
#     obs: np.ndarray = None
#     is_first: bool = True
#     is_last: bool = False
#     to_play: int = 0
#     reward: float = 0.0
#     action: int = 0
#     discount: float = 1.0
#     legal_actions_mask: np.ndarray = np.array(1)

#     # planner output
#     root_value: float = 0
#     action_probs: np.ndarray = np.array(0.0)
#     mcts_root: Optional[Any] = None

#     # player inputs
#     stacked_frames: np.ndarray = np.array(0)
#     policy_feed: Optional[PolicyFeed] = None

#     input_buffer: tuple = field(default_factory=tuple)
#     output_buffer: tuple = field(default_factory=tuple)

#     signals: Dict[str, bool] = field(default_factory=lambda: {"exit": False})
