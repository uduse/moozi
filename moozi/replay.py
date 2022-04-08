import asyncio
import collections
import random
from dataclasses import dataclass, field
from enum import EnumMeta
from typing import Deque, List, NamedTuple

import chex
import numpy as np
import ray
import tensorflow as tf
from acme.utils import tree_utils
from loguru import logger
from nptyping import NDArray

import moozi as mz
from moozi.core import Config, TrajectorySample, TrainTarget
from moozi.logging import LogDatum, LogScalar


@dataclass(repr=False)
class ReplayBuffer:
    # TODO: pre-fetch with async
    max_size: int = 1_000_000
    min_size: int = 1_000
    num_unroll_steps: int = 5
    num_td_steps: int = 5
    num_stacked_frames: int = 4

    store: Deque[TrajectorySample] = field(init=False)

    prefetch_size: int = 1_000
    _prefetch_buffer: List[TrajectorySample] = field(default_factory=list)

    @staticmethod
    def from_config(config: Config, remote: bool = False):
        kwargs = dict(
            max_size=config.replay_max_size,
            min_size=config.replay_min_size,
            num_unroll_steps=config.num_unroll_steps,
            num_td_steps=config.num_td_steps,
            num_stacked_frames=config.num_stacked_frames,
        )
        if remote:
            return ray.remote(ReplayBuffer).remote(**kwargs)
        else:
            return ReplayBuffer(**kwargs)

    def __post_init__(self):
        self.store = collections.deque(maxlen=self.max_size)
        asyncio.create_task(self.launch_prefetcher())

    async def add_samples(self, samples: List[TrajectorySample]):
        self.store.extend(samples)
        logger.debug(f"Replay buffer size: {self.size()}")
        return self.size()

    async def get_batch(self, batch_size=1):
        if not self.store:
            logger.warning("Empty replay buffer")
            return []
        else:
            if self.is_started():
                while True:
                    if len(self._prefetch_buffer) >= batch_size:
                        return await self.take_batch_from_prefetch_buffer(batch_size)
                    else:
                        # wait for prefetcher
                        logger.warning("Waiting for prefetcher")
                        await asyncio.sleep(5)
            else:
                return []

    async def take_batch_from_prefetch_buffer(self, batch_size: int):
        if len(self._prefetch_buffer) >= batch_size:
            batch = self._prefetch_buffer.copy()
            self._prefetch_buffer = self._prefetch_buffer[batch_size:]
            return tree_utils.stack_sequence_fields(batch)
        else:
            await asyncio.sleep(5)

    def sample_target_from_store(self) -> TrajectorySample:
        traj = random.choice(self.store)
        random_start_idx = random.randrange(len(traj.last_reward))
        target = make_target_from_traj(
            traj,
            start_idx=random_start_idx,
            discount=1.0,
            num_unroll_steps=self.num_unroll_steps,
            num_td_steps=self.num_td_steps,
            num_stacked_frames=self.num_stacked_frames,
        )
        return target

    async def launch_prefetcher(self):
        while len(self._prefetch_buffer) <= self.prefetch_size:
            if self.is_started():
                target = self.sample_target_from_store()
                self._prefetch_buffer.append(target)
            else:
                logger.warning("Waiting for replay buffer to be started")
                await asyncio.sleep(5)

    def is_started(self):
        return self.size() >= self.min_size

    def __len__(self):
        return self.size()

    def size(self):
        return len(self.store)

    def get_stats(self):
        return dict(size=self.size())


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
