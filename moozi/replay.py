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

from acme.wrappers import open_spiel_wrapper
from acme.adders.reverb import DEFAULT_PRIORITY_TABLE, ReverbAdder, base
from acme.adders.reverb import utils as acme_reverb_utils
from acme.utils import tree_utils as acme_tree_utils

import moozi as mz


def spec_like_to_tensor_spec(paths: Iterable[str], spec: specs.Array):
    return tf.TensorSpec.from_spec(spec, name="/".join(str(p) for p in paths))


class Observation(NamedTuple):
    frame: chex.Array
    reward: np.float32
    legal_actions_mask: chex.Array
    is_first: np.bool
    is_last: np.bool

    @staticmethod
    def signature(env_spec: specs.EnvironmentSpec):
        return Observation(
            frame=env_spec.observations.observation,
            reward=env_spec.rewards,
            legal_actions_mask=specs.Array(
                shape=(env_spec.actions.num_values,), dtype=np.bool
            ),
            is_first=specs.Array(shape=(), dtype=np.bool),
            is_last=specs.Array(shape=(), dtype=np.bool),
        )

    @staticmethod
    def from_env_timestep(timestep: dm_env.TimeStep):
        if isinstance(timestep.observation[0], open_spiel_wrapper.OLT):
            frame = timestep.observation[0].observation.astype(np.float32)
            if timestep.reward is None:
                reward = np.float32(0)
            else:
                reward = np.float32(timestep.reward).squeeze()
            legal_actions_mask = timestep.observation[0].legal_actions.astype(np.bool)
            is_first = np.bool(timestep.first())
            is_last = np.bool(timestep.last())
            return Observation(frame, reward, legal_actions_mask, is_first, is_last)
        else:
            raise NotImplementedError


class Reflection(NamedTuple):
    action: np.int32
    root_value: np.float32
    child_visits: List[float]

    @staticmethod
    def signature(env_spec: specs.EnvironmentSpec):
        return Reflection(
            action=specs.Array(shape=(), dtype=np.int32),
            root_value=specs.Array(shape=(), dtype=np.float32),
            child_visits=specs.Array(
                shape=(env_spec.actions.num_values,), dtype=np.float32
            ),
        )


def _to_tensor_spec(path, spec):
    return tf.TensorSpec.from_spec(spec, path[0])


def _add_time_dim(spec):
    return tf.TensorSpec(shape=(None, *spec.shape), dtype=spec.dtype, name=spec.name)


def make_signature(env_spec: specs.EnvironmentSpec):
    specs = {
        **Observation.signature(env_spec)._asdict(),
        **Reflection.signature(env_spec)._asdict(),
    }

    specs = tree.map_structure_with_path(_to_tensor_spec, specs)
    return tree.map_structure(_add_time_dim, specs)


class MooZiAdder(ReverbAdder):
    def __init__(
        self,
        client: reverb.Client,
        #
        # num_unroll_steps: int = 5,
        # num_stacked_frames: int = 1,
        # num_td_steps: int = 1000,
        # discount: float = 1,
        #
        delta_encoded: bool = False,
        max_inflight_items: int = 1,
    ):

        self._client = client
        # self._num_unroll_steps = num_unroll_steps
        # self._num_td_steps = num_td_steps
        # self._num_stacked_frames = num_stacked_frames
        # self._discount = tree.map_structure(np.float32, discount)

        # according to the pseudocode, 500 is roughly enough for board games
        max_sequence_length = 500
        super().__init__(
            client=client,
            max_sequence_length=max_sequence_length,
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
            padding_step = tree.map_structure(np.zeros_like, last_reflection)._asdict()
            self._writer.append(padding_step, partial_step=False)
            self._write_last()
            self.reset()

    def _write(self):
        # This adder only writes at the end of the episode, see _write_last()
        pass

    def _write_last(self):
        trajectory = tree.map_structure(lambda x: x[:], self._writer.history)
        self._writer.create_item(DEFAULT_PRIORITY_TABLE, 1, trajectory)
