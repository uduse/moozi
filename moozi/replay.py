import collections
import random
from dataclasses import dataclass, field
from enum import EnumMeta
from typing import Deque, List, NamedTuple
import chex

import numpy as np
import tensorflow as tf
from acme.utils import tree_utils
from nptyping import NDArray

import moozi as mz
from moozi.logging import LoggerDatum, LoggerDatumScalar


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
            root_value=np.asarray(self.root_value, dtype=np.float32),
            action_probs=np.asarray(self.action_probs, dtype=np.float32),
            action=np.asarray(self.action, dtype=np.int32),
        )


# Trajectory is a StepSample with stacked values
TrajectorySample = StepSample
# class TrajectorySample(StepSample):
#     pass


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
            random_start_idx = random.randrange(len(traj.last_reward))
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

    def get_stats(self):
        return dict(size=self.size())

    def get_logger_data(self) -> List[LoggerDatum]:
        return [LoggerDatumScalar("replay_buffer_size", self.size())]


def make_target_from_traj(
    sample: TrajectorySample,
    start_idx,
    discount,
    num_unroll_steps,
    num_td_steps,
    num_stacked_frames,
):
    # assert not batched
    assert len(sample.last_reward.shape) == 1

    last_step_idx = sample.is_last.argmax()

    stacked_frames = _get_stacked_frames(sample, start_idx, num_stacked_frames)

    # unroll
    unrolled_data = []
    # root_player = sample.to_play[start_idx]
    for curr_idx in range(start_idx, start_idx + num_unroll_steps + 1):
        value = _get_value(sample, curr_idx, last_step_idx, num_td_steps, discount)
        last_reward = _get_last_reward(sample, start_idx, curr_idx, last_step_idx)
        action_probs = _get_action_probs(sample, curr_idx, last_step_idx)
        unrolled_data.append((value, last_reward, action_probs))

    unrolled_data_stacked = tree_utils.stack_sequence_fields(unrolled_data)

    action = _get_action(sample, start_idx, num_unroll_steps)

    return TrainTarget(
        stacked_frames=stacked_frames,
        action=action,
        value=unrolled_data_stacked[0],
        last_reward=unrolled_data_stacked[1],
        action_probs=unrolled_data_stacked[2],
    )


def _get_action(sample: TrajectorySample, start_idx, num_unroll_steps):
    action = sample.action[start_idx : start_idx + num_unroll_steps]
    num_actions_to_pad = num_unroll_steps - action.size
    if num_actions_to_pad > 0:
        action = np.concatenate((action, np.full(num_actions_to_pad, -1)))
    return action


def _get_stacked_frames(sample: TrajectorySample, start_idx, num_stacked_frames):
    _, height, width, num_channels = sample.frame.shape
    frame_idx_lower = max(start_idx - num_stacked_frames + 1, 0)
    stacked_frames = sample.frame[frame_idx_lower : start_idx + 1]
    num_frames_to_pad = num_stacked_frames - stacked_frames.shape[0]
    stacked_frames = stacked_frames.reshape((height, width, -1))
    if num_frames_to_pad > 0:
        padding = np.zeros((height, width, num_frames_to_pad * num_channels))
        stacked_frames = np.concatenate([padding, stacked_frames], axis=-1)
    chex.assert_shape(
        stacked_frames, (height, width, num_stacked_frames * num_channels)
    )
    return stacked_frames


def _get_value(
    sample: TrajectorySample,
    curr_idx,
    last_step_idx,
    num_td_steps,
    discount,
    # root_player,
):
    # value is computed based on current player instead of root player
    if curr_idx >= last_step_idx:
        return 0

    bootstrap_idx = curr_idx + num_td_steps

    accumulated_reward = _get_accumulated_reward(
        sample, curr_idx, discount, bootstrap_idx
    )
    bootstrap_value = _get_bootstrap_value(
        sample, last_step_idx, num_td_steps, discount, bootstrap_idx
    )

    return accumulated_reward + bootstrap_value


def _get_bootstrap_value(
    sample: TrajectorySample,
    last_step_idx,
    num_td_steps,
    discount,
    bootstrap_idx,
) -> float:
    if bootstrap_idx <= last_step_idx:
        value = sample.root_value[bootstrap_idx] * (discount ** num_td_steps)
        if sample.to_play[bootstrap_idx] != mz.BASE_PLAYER:
            return -value
        else:
            return value
    else:
        return 0.0


def _get_accumulated_reward(
    sample: TrajectorySample, curr_idx, discount, bootstrap_idx
) -> float:
    reward_sum = 0.0
    last_rewards = sample.last_reward[curr_idx + 1 : bootstrap_idx + 1]
    players_of_last_rewards = sample.to_play[curr_idx:bootstrap_idx]
    for i, (last_rewrad, player) in enumerate(
        zip(last_rewards, players_of_last_rewards)
    ):
        discounted_reward = last_rewrad * (discount ** i)
        # if player == mz.BASE_PLAYER:
        #     reward_sum += discounted_reward
        # else:
        #     reward_sum -= discounted_reward
        reward_sum += discounted_reward
    return reward_sum


def _get_last_reward(sample: TrajectorySample, start_idx, curr_idx, last_step_idx):
    if curr_idx == start_idx:
        return 0
    elif curr_idx <= last_step_idx:
        # TODO: is this correct?
        player_of_reward = sample.to_play[curr_idx - 1]
        # if player_of_reward == mz.BASE_PLAYER:
        #     return sample.last_reward[curr_idx]
        # else:
        #     return -sample.last_reward[curr_idx]
        return sample.last_reward[curr_idx]
    else:
        return 0


def _get_action_probs(sample: TrajectorySample, curr_idx, last_step_idx):
    if curr_idx <= last_step_idx:
        return sample.action_probs[curr_idx]
    else:
        return np.ones_like(sample.action_probs[0]) / len(sample.action_probs[0])
