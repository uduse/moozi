import typing

import copy
import chex
import numpy as np
import reverb
import tree
from acme import specs as acme_specs
from acme import types as acme_types
from acme.adders import reverb as acme_reverb
from acme.adders.reverb import base
from acme.adders.reverb import utils as acme_reverb_utils
from acme.utils import tree_utils as acme_tree_utils
import moozi as mz


import abc
from typing import Callable, Iterable, Mapping, NamedTuple, Optional, Union, Tuple

from absl import logging
from acme import specs
from acme import types
from acme.adders import base
import dm_env
import numpy as np
import reverb
import tensorflow as tf
import tree

StartOfEpisodeType = Union[bool, specs.Array, tf.Tensor, tf.TensorSpec, Tuple[()]]


class PriorityFnInput(NamedTuple):
    """The input to a priority function consisting of stacked steps."""

    observations: types.NestedArray
    actions: types.NestedArray
    rewards: types.NestedArray
    discounts: types.NestedArray
    start_of_episode: types.NestedArray
    extras: types.NestedArray


# Define the type of a priority function and the mapping from table to function.
PriorityFn = Callable[["PriorityFnInput"], float]
PriorityFnMapping = Mapping[str, Optional[PriorityFn]]


def spec_like_to_tensor_spec(paths: Iterable[str], spec: specs.Array):
    return tf.TensorSpec.from_spec(spec, name="/".join(str(p) for p in paths))


class ReverbAdder(object):
    """Base class for Reverb adders."""

    def __init__(
        self,
        client: reverb.Client,
        max_sequence_length: int,
        max_in_flight_items: int,
        delta_encoded: bool = False,
    ):
        """Initialize a ReverbAdder instance.

        Args:
          client: A client to the Reverb backend.
          max_sequence_length: The maximum length of sequences (corresponding to the
            number of observations) that can be added to replay.
          max_in_flight_items: The maximum number of items allowed to be "in flight"
            at the same time. See `block_until_num_items` in
            `reverb.TrajectoryWriter.flush` for more info.
          delta_encoded: If `True` (False by default) enables delta encoding, see
            `Client` for more information.
        """

        self._client = client
        self._max_sequence_length = max_sequence_length
        self._delta_encoded = delta_encoded
        self._max_in_flight_items = max_in_flight_items

        # This is exposed as the _writer property in such a way that it will create
        # a new writer automatically whenever the internal __writer is None. Users
        # should ONLY ever interact with self._writer.
        self.__writer = None
        # Every time a new writer is created, it must fetch the signature from the
        # Reverb server. If this is set too low it can crash the adders in a
        # distributed setup where the replay may take a while to spin up.
        self._get_signature_timeout_ms = 300_000

    def __del__(self):
        if self.__writer is not None:
            timeout_ms = 10_000
            # Try flush all appended data before closing to avoid loss of experience.
            try:
                self.__writer.flush(self._max_in_flight_items, timeout_ms=timeout_ms)
            except reverb.DeadlineExceededError as e:
                logging.error(
                    "Timeout (%d ms) exceeded when flushing the writer before "
                    "deleting it. Caught Reverb exception: %s",
                    timeout_ms,
                    str(e),
                )
            self.__writer.close()

    @property
    def _writer(self) -> reverb.TrajectoryWriter:
        if self.__writer is None:
            self.__writer = self._client.trajectory_writer(
                num_keep_alive_refs=self._max_sequence_length,
                get_signature_timeout_ms=self._get_signature_timeout_ms,
            )
        return self.__writer

    def add_priority_table(self, table_name: str, priority_fn: Optional[PriorityFn]):
        if table_name in self._priority_fns:
            raise ValueError(
                f"A priority function already exists for {table_name}. "
                f'Existing tables: {", ".join(self._priority_fns.keys())}.'
            )
        self._priority_fns[table_name] = priority_fn

    def reset(self, timeout_ms: Optional[int] = None):
        """Resets the adder's buffer."""
        if self.__writer:
            # Flush all appended data and clear the buffers.
            self.__writer.end_episode(clear_buffers=True, timeout_ms=timeout_ms)

    def add_first(self, timestep: dm_env.TimeStep):
        """Record the first observation of a trajectory."""
        if not timestep.first():
            raise ValueError(
                "adder.add_first with an initial timestep (i.e. one for "
                "which timestep.first() is True"
            )

        # Record the next observation but leave the history buffer row open by
        # passing `partial_step=True`.
        self._writer.append(
            dict(observation=timestep.observation, start_of_episode=timestep.first()),
            partial_step=True,
        )

    def add(
        self,
        action: types.NestedArray,
        next_timestep: dm_env.TimeStep,
        extras: types.NestedArray = (),
    ):
        """Record an action and the following timestep."""

        try:
            history = self._writer.history
        except RuntimeError:
            raise ValueError("adder.add_first must be called before adder.add.")

        # Add the timestep to the buffer.
        current_step = dict(
            # Observation was passed at the previous add call.
            action=action,
            reward=next_timestep.reward,
            discount=next_timestep.discount,
            # Start of episode indicator was passed at the previous add call.
            **({"extras": extras} if extras else {}),
        )
        self._writer.append(current_step)

        # Record the next observation and write.
        self._writer.append(
            dict(
                observation=next_timestep.observation,
                start_of_episode=next_timestep.first(),
            ),
            partial_step=True,
        )
        self._write()

        if next_timestep.last():
            # Complete the row by appending zeros to remaining open fields.
            # TODO(b/183945808): remove this when fields are no longer expected to be
            # of equal length on the learner side.
            dummy_step = tree.map_structure(np.zeros_like, current_step)
            self._writer.append(dummy_step)
            self._write_last()
            self.reset()

    @abc.abstractmethod
    def _write(self):
        """Write data to replay from the buffer."""

    @abc.abstractmethod
    def _write_last(self):
        """Write data to replay from the buffer."""


class MooZiAdder(acme_reverb.ReverbAdder):
    def __init__(
        self,
        client: reverb.Client,
        num_unroll_steps: int = 5,
        num_stacked_frames: int = 1,
        num_td_steps: int = 1000,
        discount: float = 1,
    ):

        self._client = client
        self._num_unroll_steps = num_unroll_steps
        self._num_td_steps = num_td_steps
        self._num_stacked_frames = num_stacked_frames
        self._discount = tree.map_structure(np.float32, discount)

        # according to the pseudocode, 500 is roughly enough for board games
        max_sequence_length = 500
        super().__init__(
            client=client,
            max_sequence_length=max_sequence_length,
            max_in_flight_items=1,
        )

    def _write(self):
        # This adder only writes at the end of the episode, see _write_last()
        pass

    def _write_last(self):
        trajectory = tree.map_structure(lambda x: x[:], self._writer.history)
        trajectory = base.Trajectory(**trajectory)
        print(trajectory)
        self._writer.create_item(mz.utils.DEFAULT_REPLAY_TABLE_NAME, 1, trajectory)


def make_signature(
    env_spec: acme_specs.EnvironmentSpec, num_unroll_steps: int, num_stacked_frames: int
):
    orig_obs_spec = env_spec.observations.observation
    stacked_frames_shape = (num_stacked_frames,) + orig_obs_spec.shape
    observations_spec = env_spec.observations._replace(
        observation=orig_obs_spec.replace(shape=stacked_frames_shape)
    )
    last_rewards_spec = env_spec.rewards.replace(
        shape=(num_unroll_steps + 1,) + env_spec.rewards.shape, dtype=float
    )
    child_visits_spec = acme_specs.Array(
        shape=(num_unroll_steps + 1, env_spec.actions.num_values), dtype=float
    )
    values_spec = acme_specs.Array(shape=(num_unroll_steps + 1,), dtype=float)
    actions_spec = acme_specs.BoundedArray(
        shape=(num_unroll_steps + 1,),
        dtype=int,
        minimum=0,
        maximum=env_spec.actions.num_values,
    )

    train_target_spec = mz.utils.MooZiTrainTarget(
        observations=observations_spec,
        actions=actions_spec,
        child_visits=child_visits_spec,
        last_rewards=last_rewards_spec,
        values=values_spec,
    )

    return tree.map_structure_with_path(
        base.spec_like_to_tensor_spec, train_target_spec
    )
