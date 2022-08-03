import cloudpickle
from pathlib import Path
from threading import Thread
import collections
import random
from dataclasses import dataclass, field
import time
from typing import Any, Deque, List, Optional, Sequence, Tuple, TypeVar, Union
from queue import Queue, Empty

import chex
import jax
import numpy as np
import ray
from acme.utils import tree_utils
from loguru import logger

import moozi as mz
from moozi.core import TrajectorySample, TrainTarget
from moozi.laws import make_batch_stacker, make_stacker
from moozi.logging import LogDatum, describe_np_array
from moozi.nn.training import make_target_from_traj


@dataclass
class ReplayEntry:
    payload: Any
    priority: float = 1.0
    num_sampled: int = 0
    from_env: bool = True
    freshness: float = 1.0


@dataclass(repr=False)
class ReplayBuffer:
    max_trajs_size: int = 1_000_000
    max_targets_size: int = 1_000_000
    sampling_strategy: str = "uniform"
    ranking_base: float = 1.0
    use_remote: bool = False

    num_unroll_steps: int = 5
    num_td_steps: int = 5
    num_stacked_frames: int = 4
    discount: float = 0.997
    decay: float = 0.9
    save_dir: str = "./replay/"
    name: str = "replay"

    _num_env_frames_added: int = 0
    _num_targets_created: int = 0
    _trajs: Deque[ReplayEntry] = field(init=False)
    _train_targets: Deque[ReplayEntry] = field(init=False)
    _value_diffs: Deque[float] = field(init=False)
    _root_n_step_return: Deque[float] = field(init=False)
    _root_value: Deque[float] = field(init=False)
    _episode_return: Deque[float] = field(init=False)
    _epoch: int = 0
    _steps: int = 0

    def __post_init__(self):
        self._trajs = collections.deque(maxlen=self.max_trajs_size)
        self._train_targets = collections.deque(maxlen=self.max_targets_size)
        self._value_diffs = collections.deque(maxlen=1024)
        self._root_n_step_return = collections.deque(maxlen=1024)
        self._root_value = collections.deque(maxlen=1024)
        self._episode_return = collections.deque(maxlen=256)
        self._tb_logger = mz.logging.JAXBoardLoggerV2(
            name=self.name,
            time_delta=30,
            log_dir="./tb/",
        )
        logger.remove()
        logger.add(f"logs/replay/{self.name}.log", level="DEBUG")
        logger.info(f"Replay buffer created, {vars(self)}")
        self._flog = logger

    def add_trajs(self, trajs: List[TrajectorySample], from_env: bool = True):
        if from_env:
            self._trajs.extend([ReplayEntry(traj) for traj in trajs])
            for traj in trajs:
                self._num_env_frames_added += traj.frame.shape[0]
        self._flog.debug(f"Added {len(trajs)} trajs, {from_env=}")
        num_processed = self._process_trajs(trajs, from_env=from_env)
        self._flog.debug(f"Processed into {num_processed} targets")
        self._flog.debug(str(self.get_stats()))
        self._steps += 1
        return self.get_trajs_size()

    def _process_trajs(self, trajs: List[TrajectorySample], from_env: bool) -> int:
        num_processed = 0
        for traj in trajs:
            chex.assert_rank(traj.last_reward, 1)
            traj_len = traj.action.shape[0]
            for i in range(traj_len):
                num_td_step = self.num_td_steps
                target = make_target_from_traj(
                    traj,
                    start_idx=i,
                    discount=self.discount,
                    num_unroll_steps=self.num_unroll_steps,
                    num_td_steps=num_td_step,
                    num_stacked_frames=self.num_stacked_frames,
                )
                value_diff = np.abs(target.n_step_return[0] - target.root_value[0])
                self._root_n_step_return.append(target.n_step_return[0])
                self._root_value.append(target.root_value[0])
                self._value_diffs.append(value_diff)
                self._train_targets.append(
                    ReplayEntry(target, priority=value_diff, from_env=from_env)
                )
                num_processed += 1
                self._num_targets_created += 1
            self._episode_return.append(np.sum(traj.last_reward))
        return num_processed

    def get_train_targets_batch(self, batch_size: int) -> TrainTarget:
        # if self.get_targets_size() < self.min_size:
        #     raise ValueError("Not enough samples in replay buffer.")
        if len(self._train_targets) == 0:
            logger.error(f"No train targets available")
            raise ValueError("No train targets available")
        entries = self._sample_targets(batch_size)
        train_targets = [entry.payload for entry in entries]
        logger.debug(f"Returning {len(train_targets)} train targets")
        return tree_utils.stack_sequence_fields(train_targets)

    def pop_train_targets_batch(self, batch_size: int) -> TrainTarget:
        if len(self._train_targets) == 0:
            logger.error(f"No train targets available")
            raise ValueError("No train targets available")
        entries = []
        for _ in range(batch_size):
            entries.append(self._train_targets.popleft())
        train_targets = [entry.payload for entry in entries]
        logger.debug(f"Popping {len(train_targets)} train targets")
        return tree_utils.stack_sequence_fields(train_targets)

    def get_trajs_batch(self, batch_size: int) -> Tuple[TrajectorySample, ...]:
        if len(self._trajs) == 0:
            logger.error(f"No trajs available")
            raise ValueError("No trajs available")
        entries = np.random.choice(self._trajs, size=batch_size)
        trajs = [entry.payload for entry in entries]
        logger.debug(f"Returning {len(trajs)} trajs")
        return tuple(trajs)

    def get_trajs_size(self):
        return len(self._trajs)

    def get_targets_size(self):
        return len(self._train_targets)

    def get_num_targets_created(self):
        return self._num_targets_created

    def get_stats(self):
        ret = {
            "trajs_size": self.get_trajs_size(),
            "targets_size": self.get_targets_size(),
        }
        if self._value_diffs:
            ret.update(describe_np_array(self._value_diffs, "value_diff"))
        if self._root_n_step_return:
            ret.update(describe_np_array(self._root_n_step_return, "root_n_step_return"))
        if self._root_value:
            ret.update(describe_np_array(self._root_value, "root_value"))
        if self._episode_return:
            ret.update(describe_np_array(self._episode_return, "episode_return"))
        num_sampled = [item.num_sampled for item in self._train_targets]
        if num_sampled:
            ret.update(describe_np_array(num_sampled, "num_sampled"))
        ret["reanalyze_ratio"] = sum(
            [(not item.from_env) for item in self._train_targets]
        ) / max(len(self._train_targets), 1)
        ret["num_env_frames_added"] = self._num_env_frames_added
        return {'replay/' + key: val for key, val in ret.items()}

    def apply_decay(self):
        for target in self._train_targets:
            target.freshness *= self.decay
        for traj in self._trajs:
            traj.freshness *= self.decay

    def _sample_targets(self, batch_size: int) -> List[ReplayEntry]:
        if self.sampling_strategy == "uniform":
            weights = np.ones(len(self._train_targets))
        elif self.sampling_strategy == "ranking":
            weights = self._compute_ranking_weights()
        elif self.sampling_strategy == "value_diff":
            weights = self._compute_value_diff_weights()
        elif self.sampling_strategy == "hybrid":
            ranking_weights = self._compute_ranking_weights()
            freq_weights = self._compute_freq_weights()
            weights = np.log(ranking_weights) * np.log(freq_weights)
            weights /= np.sum(weights)
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")

        if self.decay < 1.0:
            weights *= np.array([e.freshness for e in self._train_targets])
            weights /= np.sum(weights)

        indices = np.arange(len(self._train_targets))
        batch_indices = np.random.choice(indices, size=batch_size, p=weights)
        batch = [self._train_targets[i] for i in batch_indices]
        is_ratio = (1 / weights[batch_indices]) / weights.size
        for i, target in enumerate(batch):
            target.payload = target.payload._replace(
                importance_sampling_ratio=is_ratio[i]
            )
        for item in batch:
            item.num_sampled += 1
        return batch

    def _compute_freq_weights(self):
        counts = np.array([item.num_sampled for item in self._train_targets])
        weights = 1 / (counts + 1)
        weights /= np.sum(weights)
        return weights

    def _compute_ranking_weights(self):
        priorities = np.array([item.priority for item in self._train_targets])
        ranks = np.argsort(-priorities)
        weights = 1 / (ranks + self.ranking_base)
        weights /= np.sum(weights)
        return weights

    def _compute_value_diff_weights(self):
        priorities = np.array([item.priority for item in self._train_targets])
        weights = priorities
        weights += 0.1
        weights /= np.sum(weights)
        return weights

    def save(self):
        path = (
            Path(self.save_dir).expanduser()
            / f"{self.name}_{self._num_targets_created}.pkl"
        )
        path = str(path)
        logger.info(f"saving replays to {path}")
        with open(path, "wb") as f:
            cloudpickle.dump(self.__dict__, f)

    def restore(self, path):
        logger.info(f"restoring replays from {path}")
        with open(path, "rb") as f:
            self.__dict__ = cloudpickle.load(f)

    def log_tensorboard(self):
        log_datum = LogDatum.from_any(self.get_stats())
        self._tb_logger.write(log_datum, step=self._steps)
