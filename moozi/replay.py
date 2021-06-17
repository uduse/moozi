import copy
from os import stat
from typing import Iterable, List, NamedTuple

import chex
import dm_env
import numpy as np
import reverb
import tensorflow as tf
import tree
from acme import specs
from acme import types as acme_types
from acme.adders.reverb import DEFAULT_PRIORITY_TABLE, ReverbAdder, base
from acme.adders.reverb import utils as acme_reverb_utils
from acme.utils import tree_utils as acme_tree_utils
from acme.wrappers import open_spiel_wrapper
from reverb.trajectory_writer import TrajectoryColumn

import moozi as mz

# def spec_like_to_tensor_spec(paths: Iterable[str], spec: specs.Array):
#     return tf.TensorSpec.from_spec(spec, name="/".join(str(p) for p in paths))


class Observation(NamedTuple):
    frame: chex.Array
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
    action: np.int32
    root_value: np.float32
    child_visits: List[float]


def _to_tensor_spec(path, spec):
    return tf.TensorSpec.from_spec(spec, path[0])


def _prefix_dim(spec, size):
    return spec.replace(shape=(size,) + spec.shape)


def make_signature(
    env_spec: specs.EnvironmentSpec,
    num_unroll_steps,
    num_stacked_frames,
    num_td_steps,
    max_episode_length,
):

    reward_size = max(num_unroll_steps + num_td_steps + 1, max_episode_length)
    obs_signature = Observation(
        frame=_prefix_dim(env_spec.observations.observation, num_stacked_frames),
        reward=_prefix_dim(env_spec.rewards, reward_size),
        # legal_actions_mask=specs.Array(
        #     shape=(env_spec.actions.num_values,), dtype=np.bool
        # ),
        is_first=specs.Array(shape=(num_unroll_steps + 1,), dtype=np.bool),
        is_last=specs.Array(shape=(num_unroll_steps + 1,), dtype=np.bool),
    )

    bootstrap_room_left = max(max_episode_length - num_td_steps - 1, 0)
    root_value_size = min(bootstrap_room_left, num_td_steps + 1)
    root_value_shape = (root_value_size,) if root_value_size else (1,)
    ref_signature = Reflection(
        action=specs.Array(shape=(num_unroll_steps,), dtype=np.int32),
        root_value=specs.Array(shape=root_value_shape, dtype=np.float32),
        child_visits=specs.Array(
            shape=(num_unroll_steps + 1, env_spec.actions.num_values), dtype=np.float32
        ),
    )

    merged_signature = {**obs_signature._asdict(), **ref_signature._asdict()}
    return tree.map_structure(tf.TensorSpec.from_spec, merged_signature)


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
        # self._discount = tree.map_structure(np.float32, discount)

        self._max_episode_length = max_episode_length
        super().__init__(
            client=client,
            max_sequence_length=self._max_episode_length
            + 1,  # extra place for dummy padding
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

            # create dummy values for referencing
            dummy_padding_step = {
                **tree.map_structure(np.zeros_like, last_reflection)._asdict(),
                **tree.map_structure(np.zeros_like, next_observation)._asdict(),
            }
            self._writer.append(dummy_padding_step, partial_step=False)

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

        history = self._writer.history
        last_step_idx = self._writer.history["is_last"][:].numpy().argmax()
        dummy_step_idx = -1

        assert last_step_idx == len(self._writer.history["is_last"]) - 2

        for slice_idx in range(last_step_idx):
            frame_slice = history["frame"][: slice_idx + 1]
            num_frames_to_pad = self._num_stacked_frames - len(frame_slice)
            if num_frames_to_pad > 0:
                dummy_frame = history["frame"][dummy_step_idx]
                dummy_references = [dummy_frame for _ in range(num_frames_to_pad)]
                frame_slice = TrajectoryColumn(dummy_references + list(frame_slice))
            assert len(frame_slice) == self._num_stacked_frames

            episode_stop = last_step_idx + 1
            td_stop = slice_idx + self._num_td_steps + 1
            reward_slice_stop = max(episode_stop, td_stop)
            reward_slice = history["reward"][slice_idx:reward_slice_stop]

            episode_stop = last_step_idx + 1
            unroll_stop = slice_idx + self._num_unroll_steps
            action_slice_stop = max(episode_stop, unroll_stop)
            action_slice = history["action"][slice_idx:action_slice_stop]

            full_slice = {
                "frame": frame_slice,
                "reward": reward_slice,
                "action": action_slice,
            }

            # target_slice = tree.map_structure(lambda x: x[start_idx:end_idx], history)
            # print(target_slice)
            self._writer.create_item(DEFAULT_PRIORITY_TABLE, 1, full_slice)

        # method #2
        # padding_step = tree.map_structure(
        #     lambda x: np.zeros(x[-1].shape, x[-1].dtype), self._writer.history
        # )
        # while self._writer.episode_steps < self._max_sequence_length:
        #     self._writer.append(padding_step)
        # trajectory = tree.map_structure(lambda x: x[:], self._writer.history)
        # self._writer.create_item(DEFAULT_PRIORITY_TABLE, 1, trajectory)
