import asyncio
from threading import Thread
import collections
import random
from dataclasses import dataclass, field
import time
from typing import Any, Deque, List, Optional, Sequence, TypeVar, Union
from queue import Queue, Empty

import chex
import numpy as np
import ray
from acme.utils import tree_utils
from loguru import logger

import moozi as mz
from moozi.core import Config, TrajectorySample, TrainTarget


@dataclass
class ReplayEntry:
    payload: Any
    weight: float = 1.0
    num_sampled: int = 0


@dataclass(repr=False)
class ReplayBuffer:
    max_size: int = 1_000_000
    min_size: int = 1_000
    prefetch_max_size: int = 1_000
    decay: float = 0.99

    num_unroll_steps: int = 5
    num_td_steps: int = 5
    num_stacked_frames: int = 4
    discount: float = 1.0

    _trajs: Deque[ReplayEntry] = field(init=False)
    _train_targets: Deque[ReplayEntry] = field(init=False)
    _value_diffs: Deque[float] = field(init=False)

    def __post_init__(self):
        self._trajs = collections.deque(maxlen=self.max_size)
        self._train_targets = collections.deque(maxlen=self.max_size)
        self._value_diffs = collections.deque(maxlen=256)
        logger.remove()
        logger.add("logs/replay.log", level="DEBUG")
        logger.info(f"Replay buffer created, {vars(self)}")

    @staticmethod
    def from_config(config: Config, remote: bool = False):
        # TODO: have two separate sizes for train targets and trajs
        kwargs = dict(
            max_size=config.replay_max_size,
            min_size=config.replay_min_size,
            discount=config.discount,
            num_unroll_steps=config.num_unroll_steps,
            num_td_steps=config.num_td_steps,
            num_stacked_frames=config.num_stacked_frames,
            prefetch_max_size=config.replay_prefetch_max_size,
            decay=config.replay_decay,
        )
        if remote:
            return ray.remote(ReplayBuffer).remote(**kwargs)
        else:
            return ReplayBuffer(**kwargs)

    def add_trajs(self, trajs: List[TrajectorySample]):
        self._trajs.extend([ReplayEntry(traj) for traj in trajs])
        self.process_trajs(trajs)
        logger.debug(f"Added {len(trajs)} trajs to processing queue")
        logger.debug(f"Size after adding samples: {self.get_trajs_size()}")
        return self.get_trajs_size()

    def process_trajs(self, trajs: List[TrajectorySample]):
        for traj in trajs:
            for i in range(len(traj.last_reward) - 1):
                target = make_target_from_traj(
                    traj,
                    start_idx=i,
                    discount=self.discount,
                    num_unroll_steps=self.num_unroll_steps,
                    num_td_steps=self.num_td_steps,
                    num_stacked_frames=self.num_stacked_frames,
                )
                value_diff = np.abs(target.n_step_return[0] - target.root_value[0])
                target = target._replace(weight=value_diff)
                self._value_diffs.append(value_diff)
                self._train_targets.append(ReplayEntry(target, weight=value_diff))

    def get_trajs_batch(self, batch_size: int = 1) -> List[TrajectorySample]:
        if len(self._trajs) == 0:
            logger.error(f"No trajs available")
            raise ValueError("No trajs available")
        entries = self.sample_entries(self._trajs, batch_size)
        return [entry.payload for entry in entries]

    def get_train_targets_batch(self, batch_size: int = 1) -> TrainTarget:
        if len(self._train_targets) == 0:
            logger.error(f"No train targets available")
            raise ValueError("No train targets available")
        entries = self.sample_entries(self._train_targets, batch_size)
        train_targets = [entry.payload for entry in entries]
        logger.debug(f"Returning {len(train_targets)} train targets")
        return tree_utils.stack_sequence_fields(train_targets)

    def get_trajs_size(self):
        return len(self._trajs)

    async def get_stats(self):
        ret = dict(
            trajs=self.get_trajs_size(),
        )
        if self._value_diffs:
            mean_value_diff = np.mean(self._value_diffs)
            ret["mean_value_diff"] = mean_value_diff
        return ret

    @staticmethod
    def sample_entries(
        entries: List[ReplayEntry], batch_size: int
    ) -> List[ReplayEntry]:
        weights = [item.weight for item in entries]
        batch = random.choices(entries, weights=weights, k=batch_size)
        for item in batch:
            item.num_sampled += 1
        return batch

    def apply_decay(self):
        for entry in self._tras + self._train_targets:
            entry.weight *= self.decay


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
        weight=np.array(1.0),
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
