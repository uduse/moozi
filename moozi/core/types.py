import typing
from dataclasses import asdict
from typing import List, NamedTuple

import haiku as hk
import jax
import numpy as np
import optax
import tree
from flax import struct

from .utils import stack_sequence_fields_pytree


class PolicyFeed(struct.PyTreeNode):
    stacked_frames: np.ndarray
    to_play: int
    legal_actions_mask: np.ndarray
    random_key: np.ndarray


class StepSample(struct.PyTreeNode):
    frame: np.ndarray

    # last reward from the environment
    # TODO: rename last_reward to just reward
    last_reward: np.ndarray
    is_first: np.ndarray
    is_last: np.ndarray
    to_play: np.ndarray
    # TODO: make legal_actions_mask actually useful
    legal_actions_mask: np.ndarray

    # root value after the search
    root_value: np.ndarray
    action_probs: np.ndarray
    action: np.ndarray

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

    def shapes(self) -> dict:
        return tree.map_structure(lambda x: x.shape, self)


# Trajectory is a StepSample with stacked values
# e.g., StepSample.frame shape is [H, W, C]
# TrajectorySample.frame shape is [N, H, W, C]
class TrajectorySample(StepSample):
    @staticmethod
    def from_step_samples(step_samples: List[StepSample]) -> "TrajectorySample":
        stacked = stack_sequence_fields_pytree(step_samples)
        return TrajectorySample(**asdict(stacked))


class TrainTarget(NamedTuple):
    # right now we only support perfect information games
    # so stacked_frames is a history of symmetric observations
    frame: np.ndarray

    # action taken in in each step, -1 means no action taken (terminal state)
    action: np.ndarray

    # value is computed based on the player of each timestep instead of the
    # player at the first timestep as the root player
    # this means if all rewards are positive, the values are always positive too
    n_step_return: np.ndarray

    # a faithful slice of the trajectory rewards, not flipped for multi-player games
    last_reward: np.ndarray

    # action probabilities from the search result
    action_probs: np.ndarray

    # root value after the search
    root_value: np.ndarray

    # player of the target
    to_play: np.ndarray

    # importance sampling (IS) ratio
    # 1.0 if uniform sampling
    # if a sample is twice more likely to be sampled, it should have a IS ratio of 0.5
    importance_sampling_ratio: np.ndarray

    def cast(self) -> "TrainTarget":
        return TrainTarget(
            frame=np.asarray(self.frame, dtype=np.float32),
            action=np.asarray(self.action, dtype=np.int32),
            n_step_return=np.asarray(self.n_step_return, dtype=np.float32),
            last_reward=np.asarray(self.last_reward, dtype=np.float32),
            action_probs=np.asarray(self.action_probs, dtype=np.float32),
            root_value=np.asarray(self.root_value, dtype=np.float32),
            to_play=np.asarray(self.to_play, dtype=np.int32),
            importance_sampling_ratio=np.asarray(
                self.importance_sampling_ratio, dtype=np.float32
            ),
        )

    def shapes(self) -> dict:
        return tree.map_structure(lambda x: x.shape, self)


# TODO: maybe make statistics a part of training state?
class TrainingState(typing.NamedTuple):
    params: hk.Params
    target_params: hk.Params
    state: hk.State
    target_state: hk.State
    opt_state: optax.OptState
    steps: int
    rng_key: jax.random.KeyArray


BASE_PLAYER: int = 0
