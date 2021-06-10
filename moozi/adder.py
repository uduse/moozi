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
