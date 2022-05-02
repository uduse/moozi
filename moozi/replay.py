import asyncio
from threading import Thread
import collections
import random
from dataclasses import dataclass, field
import time
from typing import Any, Deque, List, Optional, Sequence, Union
from queue import Queue, Empty

import chex
import numpy as np
import ray
from acme.utils import tree_utils
from loguru import logger

import moozi as mz
from moozi.core import Config, TrajectorySample, TrainTarget


# TODO: move this into payload itself?
@dataclass
class ReplayEntry:
    payload: Any
    weight: float = 1.0


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

    _traj_entries: Deque[ReplayEntry] = field(init=False)
    # _traj_entries_to_be_processed: Queue = field(default_factory=Queue)
    # _traj_processing_thread: Thread = field(init=False)
    _train_target_entries: Deque[ReplayEntry] = field(init=False)
    # _trajs_being_processed: bool = False

    _value_diffs: Deque[float] = field(init=False)

    def __post_init__(self):
        self._traj_entries = collections.deque(maxlen=self.max_size)
        self._train_target_entries = collections.deque(maxlen=self.max_size)
        self._value_diffs = collections.deque(maxlen=100)
        logger.remove()
        logger.add("logs/replay.log", level="DEBUG")
        logger.info(f"Replay buffer created, {vars(self)}")
        # self._traj_processing_thread = Thread(target=self.process_trajs, daemon=True)
        # self._traj_processing_thread.start()

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
        self.process_trajs(trajs)
        traj_entries = [ReplayEntry(sample) for sample in trajs]
        logger.debug(f"Added {len(traj_entries)} trajs to processing queue")
        self._traj_entries.extend(traj_entries)
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
                value_diff = np.abs(
                    target.n_step_return[0] - target.root_value[0]
                ).item()
                self._value_diffs.append(value_diff)
                entry = ReplayEntry(target, weight=value_diff)
                self._train_target_entries.append(entry)

    def get_trajs_batch(self, batch_size: int = 1) -> List[TrajectorySample]:
        if len(self._traj_entries) == 0:
            logger.error(f"No trajs available")
            raise ValueError("No trajs available")
        entries = self.sample_entries(self._traj_entries, batch_size)
        train_targets: List[TrajectorySample] = [entry.payload for entry in entries]
        return train_targets

    def get_train_targets_batch(self, batch_size: int = 1) -> TrainTarget:
        if len(self._train_target_entries) == 0:
            logger.error(f"No train targets available")
            raise ValueError("No train targets available")
        entries = self.sample_entries(self._train_target_entries, batch_size)
        train_targets = [entry.payload for entry in entries]
        return tree_utils.stack_sequence_fields(train_targets)

    def get_trajs_size(self):
        return len(self._traj_entries)

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
        entries: Sequence[ReplayEntry], batch_size: int
    ) -> List[ReplayEntry]:
        weights = np.array([entry.weight for entry in entries])
        p = weights / np.sum(weights)
        return np.random.choice(entries, size=batch_size, replace=True, p=p).tolist()


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
