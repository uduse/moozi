import collections
from dataclasses import dataclass, field
import random
from typing import Deque, List, NamedTuple

import chex
import dm_env
import numpy as np
import reverb
import tensorflow as tf
import tree
from acme import datasets, specs, types
from acme.adders.reverb import DEFAULT_PRIORITY_TABLE, ReverbAdder
from acme.agents.replay import ReverbReplay
from acme.utils import tree_utils
from acme.wrappers import open_spiel_wrapper
from nptyping import NDArray

import moozi as mz


class StepSample(NamedTuple):
    frame: NDArray[np.float32]
    reward: NDArray[np.float32]
    is_first: NDArray[np.bool8]
    is_last: NDArray[np.bool8]
    action: NDArray[np.int32]
    root_value: NDArray[np.float32]
    action_probs: NDArray[np.float32]

    def cast(self) -> "StepSample":
        return StepSample(
            frame=np.asarray(self.frame, dtype=np.float32),
            reward=np.asarray(self.reward, dtype=np.float32),
            is_first=np.asarray(self.is_first, dtype=np.bool8),
            is_last=np.asarray(self.is_last, dtype=np.bool8),
            action=np.asarray(self.action, dtype=np.int32),
            root_value=np.asarray(self.root_value, dtype=np.float32),
            action_probs=np.asarray(self.action_probs, dtype=np.float32),
        )


# Trajectory is a StepSample with stacked values
class TrajectorySample(StepSample):
    pass


class TrainTarget(NamedTuple):
    stacked_frames: NDArray[np.float32]
    action: NDArray[np.int32]
    value: NDArray[np.float32]
    last_reward: NDArray[np.float32]
    action_probs: NDArray[np.float32]

    def cast(self) -> "TrainTarget":
        return TrainTarget(
            stacked_frames=np.asarray(self.stacked_frames, dtype=np.float32),
            action=np.asarray(self.action, dtype=np.int32),
            value=np.asarray(self.value, dtype=np.float32),
            last_reward=np.asarray(self.last_reward, dtype=np.float32),
            action_probs=np.asarray(self.action_probs, dtype=np.float32),
        )


@dataclass(repr=False)
class ReplayBuffer:
    # TODO: remove config here
    # TODO: pre-fetch with async
    config: mz.Config

    store: Deque[TrajectorySample] = field(init=False)

    def __post_init__(self):
        self.store = collections.deque(maxlen=self.config.replay_buffer_size)

    def add_samples(self, samples: List[TrajectorySample]):
        self.store.extend(samples)
        # logging.info(f"Replay buffer size: {self.size()}")
        return self.size()

    def get_batch(self, batch_size=1):
        if not self.store:
            raise ValueError("Empty replay buffer")

        trajs = random.choices(self.store, k=batch_size)
        batch = []
        for traj in trajs:
            random_start_idx = random.randrange(len(traj.reward))
            target = make_target_from_traj(
                traj,
                start_idx=random_start_idx,
                discount=1.0,
                num_unroll_steps=self.config.num_unroll_steps,
                num_td_steps=self.config.num_td_steps,
                num_stacked_frames=self.config.num_stacked_frames,
            )
            batch.append(target)
        return tree_utils.stack_sequence_fields(batch)

    def size(self):
        return len(self.store)


def make_target_from_traj(
    sample: TrajectorySample,
    start_idx,
    discount,
    num_unroll_steps,
    num_td_steps,
    num_stacked_frames,
):
    # assert not batched
    assert len(sample.reward.shape) == 1

    last_step_idx = sample.is_last.argmax()

    # unroll
    unrolled_data = []
    for curr_idx in range(start_idx, start_idx + num_unroll_steps + 1):
        value = _get_value(sample, curr_idx, last_step_idx, num_td_steps, discount)
        last_reward = _get_last_reward(sample, start_idx, curr_idx, last_step_idx)
        action_probs = _get_action_probs(sample, curr_idx, last_step_idx)
        unrolled_data.append((value, last_reward, action_probs))

    unrolled_data_stacked = tree_utils.stack_sequence_fields(unrolled_data)

    stacked_frames = _get_stacked_frames(sample, start_idx, num_stacked_frames)
    action = _get_action(sample, start_idx, num_unroll_steps)

    return TrainTarget(
        stacked_frames=stacked_frames,
        action=action,
        value=unrolled_data_stacked[0],
        last_reward=unrolled_data_stacked[1],
        action_probs=unrolled_data_stacked[2],
    )


def _get_action(sample, start_idx, num_unroll_steps):
    action = sample.action[start_idx : start_idx + num_unroll_steps]
    num_actions_to_pad = num_unroll_steps - action.size
    if num_actions_to_pad > 0:
        action = np.concatenate((action, np.full(num_actions_to_pad, -1)))
    return action


def _get_stacked_frames(sample, start_idx, num_stacked_frames):
    frame_idx_lower = max(start_idx - num_stacked_frames + 1, 0)
    frame = sample.frame[frame_idx_lower : start_idx + 1]
    num_frames_to_pad = num_stacked_frames - frame.shape[0]
    if num_frames_to_pad > 0:
        padding_shape = (num_frames_to_pad,) + frame.shape[1:]
        frame = np.concatenate((np.zeros(shape=padding_shape), frame))
    return frame


def _get_value(sample, curr_idx, last_step_idx, num_td_steps, discount):
    bootstrap_idx = curr_idx + num_td_steps
    if bootstrap_idx <= last_step_idx:
        value = sample.root_value[bootstrap_idx] * discount ** num_td_steps
    else:
        value = 0

    for i, reward in enumerate(sample.reward[curr_idx + 1 : bootstrap_idx + 1]):
        value += reward * discount ** i
    return value


def _get_last_reward(sample, start_idx, curr_idx, last_step_idx):
    if curr_idx == start_idx:
        return 0
    elif curr_idx <= last_step_idx:
        return sample.reward[curr_idx]
    else:
        return 0


def _get_action_probs(sample, curr_idx, last_step_idx):
    if curr_idx <= last_step_idx:
        return sample.action_probs[curr_idx]
    else:
        return np.zeros_like(sample.action_probs[0])
