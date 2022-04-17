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


@dataclass(repr=False)
class ReplayBuffer:
    max_size: int = 1_000_000
    min_size: int = 1_000
    prefetch_max_size: int = 1_000

    num_unroll_steps: int = 5
    num_td_steps: int = 5
    num_stacked_frames: int = 4
    discount: float = 1.0

    store: Deque[TrajectorySample] = field(init=False)

    _prefetch_buffer: List[TrainTarget] = field(default_factory=list)

    _last_value_diff: float = float("inf")

    @staticmethod
    def from_config(config: Config, remote: bool = False):
        kwargs = dict(
            max_size=config.replay_max_size,
            min_size=config.replay_min_size,
            discount=config.discount,
            num_unroll_steps=config.num_unroll_steps,
            num_td_steps=config.num_td_steps,
            num_stacked_frames=config.num_stacked_frames,
            prefetch_max_size=config.replay_prefetch_max_size,
        )
        if remote:
            return ray.remote(ReplayBuffer).remote(**kwargs)
        else:
            return ReplayBuffer(**kwargs)

    def __post_init__(self):
        self.store = collections.deque(maxlen=self.max_size)
        asyncio.create_task(self.launch_prefetcher())
        logger.remove()
        logger.add("logs/replay.log", level="DEBUG")
        logger.info(f"Replay buffer created, {vars(self)}")

    async def add_samples(self, samples: List[TrajectorySample]):
        logger.debug(f"Adding samples to replay buffer, size: {self.size()}")
        self.store.extend(samples)
        if samples:
            self._compute_samples_valued_diff(samples)
        logger.debug(f"Replay buffer size after adding samples: {self.size()}")
        return self.size()

    def _compute_samples_valued_diff(self, samples: List[TrajectorySample]):
        total = []
        for sample in samples:
            for i in range(len(sample.last_reward) - 1):
                target = make_target_from_traj(
                    sample,
                    start_idx=i,
                    discount=self.discount,
                    num_unroll_steps=self.num_unroll_steps,
                    num_td_steps=self.num_td_steps,
                    num_stacked_frames=self.num_stacked_frames,
                )
                total.append(np.abs(target.n_step_return - target.root_value))
        self._last_value_diff = float(np.mean(total))

    async def get_train_batch(self, batch_size: int = 1) -> TrainTarget:
        if batch_size > self.prefetch_max_size:
            raise ValueError("batch_size must be <= prefetch_max_size")

        await self.wait_until_started()

        while True:
            if len(self._prefetch_buffer) >= batch_size:
                batch = await self.take_batch_from_prefetch_buffer(batch_size)
                return batch
            else:
                logger.warning(
                    f"Waiting for prefetcher, prefetch buffer size: {len(self._prefetch_buffer)}"
                )
                await asyncio.sleep(5)

    async def get_traj_batch(self, batch_size: int = 1) -> List[TrajectorySample]:
        await self.wait_until_started()
        batch = random.choices(self.store, k=batch_size)
        logger.debug(f"Trajectory batch size: {len(batch)}")
        return batch

    async def take_batch_from_prefetch_buffer(self, batch_size: int) -> TrainTarget:
        batch = self._prefetch_buffer[:batch_size].copy()
        self._prefetch_buffer = self._prefetch_buffer[batch_size:]
        return tree_utils.stack_sequence_fields(batch)

    def _sample_train_target_from_store(self) -> TrainTarget:
        traj = random.choice(self.store)
        random_start_idx = random.randrange(len(traj.last_reward))
        target = make_target_from_traj(
            traj,
            start_idx=random_start_idx,
            discount=self.discount,
            num_unroll_steps=self.num_unroll_steps,
            num_td_steps=self.num_td_steps,
            num_stacked_frames=self.num_stacked_frames,
        )
        return target

    async def launch_prefetcher(self):
        await self.wait_until_started()

        logger.info("Prefetcher started")

        while 1:
            if len(self._prefetch_buffer) <= self.prefetch_max_size:
                for _ in range(100):
                    target = self._sample_train_target_from_store()
                    self._prefetch_buffer.append(target)
                logger.debug(f"Prefetch buffer size: {len(self._prefetch_buffer)}")
                await asyncio.sleep(0)
            else:
                await asyncio.sleep(1)

    def is_started(self):
        return self.size() >= self.min_size

    async def wait_until_started(self, delay: float = 5.0):
        while not self.is_started():
            await asyncio.sleep(delay)
            logger.debug(f"Waiting for replay buffer to start, size: {self.size()}")

    def __len__(self):
        return self.size()

    def size(self):
        return len(self.store)

    async def get_stats(self):
        return dict(replay_size=self.size(), value_diff=self._last_value_diff)


def make_target_from_traj(
    sample: TrajectorySample,
    start_idx,
    discount,
    num_unroll_steps,
    num_td_steps,
    num_stacked_frames,
) -> TrainTarget:
    # assert not batched
    assert len(sample.last_reward.shape) == 1

    last_step_idx = sample.is_last.argmax()

    stacked_frames = _get_stacked_frames(sample, start_idx, num_stacked_frames)
    action = _get_action(sample, start_idx, num_unroll_steps)

    unrolled_data = []
    for curr_idx in range(start_idx, start_idx + num_unroll_steps + 1):
        n_step_return = _get_n_step_return(
            sample, curr_idx, last_step_idx, num_td_steps, discount
        )
        last_reward = _get_last_reward(sample, start_idx, curr_idx, last_step_idx)
        action_probs = _get_action_probs(sample, curr_idx, last_step_idx)
        root_value = _get_root_value(sample, curr_idx, last_step_idx)
        unrolled_data.append((n_step_return, last_reward, action_probs, root_value))

    unrolled_data_stacked = tree_utils.stack_sequence_fields(unrolled_data)

    return TrainTarget(
        stacked_frames=stacked_frames,
        action=action,
        n_step_return=unrolled_data_stacked[0],
        last_reward=unrolled_data_stacked[1],
        action_probs=unrolled_data_stacked[2],
        root_value=unrolled_data_stacked[3],
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


def _get_n_step_return(
    sample: TrajectorySample,
    curr_idx,
    last_step_idx,
    num_td_steps,
    discount,
):
    """The observed N-step return with bootstrapping."""
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
        return sample.last_reward[curr_idx]
    else:
        return 0


def _get_action_probs(sample: TrajectorySample, curr_idx, last_step_idx):
    if curr_idx <= last_step_idx:
        return sample.action_probs[curr_idx]
    else:
        return np.ones_like(sample.action_probs[0]) / len(sample.action_probs[0])


def _get_root_value(sample: TrajectorySample, curr_idx, last_step_idx):
    if curr_idx <= last_step_idx:
        return sample.root_value[curr_idx]
    else:
        return 0.0
