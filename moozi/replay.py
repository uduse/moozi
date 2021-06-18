import copy
from os import stat
from typing import Iterable, List, NamedTuple

import chex
from nptyping import NDArray
import dm_env
import numpy as np
import reverb
import tensorflow as tf
import tree
from acme import specs
from acme import types as acme_types
from acme.adders.reverb import DEFAULT_PRIORITY_TABLE, ReverbAdder, base
from acme.adders.reverb import utils as acme_reverb_utils
from acme.utils import tree_utils
from acme.wrappers import open_spiel_wrapper
from reverb.trajectory_writer import TrajectoryColumn

import moozi as mz


class Observation(NamedTuple):
    frame: np.ndarray
    reward: np.float32
    # legal_actions_mask: chex.Array
    is_first: np.bool
    is_last: np.bool

    @staticmethod
    def from_env_timestep(timestep: dm_env.TimeStep):
        if isinstance(timestep.observation[0], open_spiel_wrapper.OLT):
            frame = timestep.observation[0].observation.astype(np.float32)
            if timestep.reward is None:
                reward = np.float32(0)
            else:
                reward = np.float32(timestep.reward).squeeze()
            # legal_actions_mask = timestep.observation[0].legal_actions.astype(np.bool)
            is_first = np.bool(timestep.first())
            is_last = np.bool(timestep.last())
            # return Observation(frame, reward, legal_actions_mask, is_first, is_last)
            return Observation(frame, reward, is_first, is_last)
        else:
            raise NotImplementedError


class Reflection(NamedTuple):
    action: np.ndarray
    root_value: np.ndarray
    child_visits: np.ndarray


def _prefix_dim(spec, size):
    return spec.replace(shape=(size,) + spec.shape)


def make_signature(env_spec: specs.EnvironmentSpec, max_episode_length):

    obs_signature = Observation(
        frame=_prefix_dim(env_spec.observations.observation, max_episode_length),
        reward=_prefix_dim(env_spec.rewards, max_episode_length),
        is_first=specs.Array(shape=(max_episode_length,), dtype=np.bool),
        is_last=specs.Array(shape=(max_episode_length,), dtype=np.bool),
    )

    ref_signature = Reflection(
        action=specs.Array(shape=(max_episode_length,), dtype=np.int32),
        root_value=specs.Array(shape=(max_episode_length,), dtype=np.float32),
        child_visits=specs.Array(
            shape=(max_episode_length, env_spec.actions.num_values), dtype=np.float32
        ),
    )

    merged_signature = {**obs_signature._asdict(), **ref_signature._asdict()}
    # return tree.map_structure(tf.TensorSpec.from_spec, merged_signature)
    return tree.map_structure_with_path(
        lambda p, v: tf.TensorSpec.from_spec(v, p[0]), merged_signature
    )


class MooZiAdder(ReverbAdder):
    def __init__(
        self,
        client: reverb.Client,
        num_unroll_steps: int = 5,
        num_stacked_frames: int = 1,
        num_td_steps: int = 1000,
        # according to the pseudocode, 500 is roughly enough for board games
        max_episode_length: int = 500,
        # discount: float = 1,
        delta_encoded: bool = False,
        max_inflight_items: int = 1,
    ):

        self._client = client
        self._num_unroll_steps = num_unroll_steps
        self._num_td_steps = num_td_steps
        self._num_stacked_frames = num_stacked_frames
        self._max_episode_length = max_episode_length

        super().__init__(
            client=client,
            max_sequence_length=self._max_episode_length,
            max_in_flight_items=max_inflight_items,
            delta_encoded=delta_encoded,
        )

    def add_first(self, observation: Observation):
        assert observation.is_first
        assert not observation.is_last
        assert np.isclose(observation.reward, 0)

        self._writer.append(observation._asdict(), partial_step=True)

    def add(self, last_reflection: Reflection, next_observation: Observation):
        try:
            _ = self._writer.history
        except RuntimeError:
            raise ValueError("adder.add_first must be called before adder.add.")

        self._writer.append(last_reflection._asdict(), partial_step=False)
        self._writer.append(next_observation._asdict(), partial_step=True)

        if next_observation.is_last:
            # pad the incomplete last step
            partial_padding_step = tree.map_structure(
                np.zeros_like, last_reflection
            )._asdict()
            self._writer.append(partial_padding_step, partial_step=False)

            self._write_last()
            self.reset()

    def _write(self):
        # This adder only writes at the end of the episode, see _write_last()
        pass

    def _write_last(self):
        r"""
        There are two ways of storing experiences for training.
        One way is to store the entire episode into the replay and create
        """
        trajectory = tree.map_structure(lambda x: x[:], self._writer.history)
        self._writer.create_item(DEFAULT_PRIORITY_TABLE, 1.0, trajectory)


class ReplaySample(NamedTuple):
    frame: NDArray[np.float32]
    reward: NDArray[np.float32]
    is_first: NDArray[np.bool]
    is_last: NDArray[np.bool]
    action: NDArray[np.int32]
    root_value: NDArray[np.float32]
    child_visits: NDArray[np.float32]

    def cast(self) -> "ReplaySample":
        return ReplaySample(
            frame=np.asarray(self.frame, dtype=np.float32),
            reward=np.asarray(self.reward, dtype=np.float32),
            is_first=np.asarray(self.is_first, dtype=np.bool),
            is_last=np.asarray(self.is_last, dtype=np.bool),
            action=np.asarray(self.action, dtype=np.int32),
            root_value=np.asarray(self.root_value, dtype=np.float32),
            child_visits=np.asarray(self.child_visits, dtype=np.float32),
        )


class TrainTarget(NamedTuple):
    frame: NDArray[np.float32]
    action: NDArray[np.int32]
    value: NDArray[np.float32]
    last_reward: NDArray[np.float32]
    child_visits: NDArray[np.float32]

    def cast(self) -> "TrainTarget":
        return TrainTarget(
            frame=np.asarray(self.frame, dtype=np.float32),
            action=np.asarray(self.action, dtype=np.int32),
            value=np.asarray(self.value, dtype=np.float32),
            last_reward=np.asarray(self.last_reward, dtype=np.float32),
            child_visits=np.asarray(self.child_visits, dtype=np.float32),
        )


def make_target(
    sample: ReplaySample,
    start_idx,
    discount,
    num_unroll_steps,
    num_td_steps,
    num_stacked_frames,
):
    # assert not batched
    assert len(sample.reward.shape) == 1

    last_step_idx = sample.is_last.argmax()
    collected = []
    for curr_idx in range(start_idx, start_idx + num_unroll_steps + 1):
        bootstrap_idx = curr_idx + num_td_steps
        if bootstrap_idx <= last_step_idx:
            value = sample.root_value[bootstrap_idx] * discount ** num_td_steps
        else:
            value = 0

        for i, reward in enumerate(sample.reward[curr_idx + 1 : bootstrap_idx + 1]):
            value += reward * discount ** i

        if curr_idx <= last_step_idx:
            last_reward = sample.reward[curr_idx]
        else:
            last_reward = 0

        frame_idx_lower = max(curr_idx - num_stacked_frames, 0)
        frame = sample.frame[frame_idx_lower : curr_idx + 1]
        num_frames_to_pad = num_stacked_frames - frame.shape[0]
        if num_frames_to_pad > 0:
            padding_shape = (num_frames_to_pad,) + frame.shape[1:]
            frame = np.concatenate((np.zeros(shape=padding_shape), frame))

        collected.append((frame, value, last_reward))

    collected_stacked = tree_utils.stack_sequence_fields(collected)
    action = sample.action[start_idx : start_idx + num_unroll_steps]
    child_visits = sample.child_visits[start_idx : start_idx + num_unroll_steps + 1]

    return TrainTarget(
        frame=collected_stacked[0],
        action=action,
        value=collected_stacked[1],
        last_reward=collected_stacked[2],
        child_visits=child_visits,
    )
