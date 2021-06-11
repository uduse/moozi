from typing import Iterable, NamedTuple

import copy
import chex
import numpy as np
import reverb
import tree
from acme import specs
from acme import types as acme_types
from acme.adders.reverb import ReverbAdder

# from acme.adders import reverb as acme_reverb
from acme.adders.reverb import base, DEFAULT_PRIORITY_TABLE
from acme.adders.reverb import utils as acme_reverb_utils
from acme.utils import tree_utils as acme_tree_utils
import moozi as mz
import tensorflow as tf


def spec_like_to_tensor_spec(paths: Iterable[str], spec: specs.Array):
    return tf.TensorSpec.from_spec(spec, name="/".join(str(p) for p in paths))


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

    def add_first(self, timestep: dm_env.TimeStep):
        return super().add_first(timestep)

    def _write(self):
        # This adder only writes at the end of the episode, see _write_last()
        pass

    def _write_last(self):
        trajectory = tree.map_structure(lambda x: x[:], self._writer.history)
        trajectory = base.Trajectory(**trajectory)
        self._writer.create_item(DEFAULT_PRIORITY_TABLE, 1, trajectory)

def make_signature(env_spec: specs.EnvironmentSpec):
    orig_obs_spec = env_spec.observations.observation
    child_visits_spec = specs.Array(
        shape=(num_unroll_steps + 1, env_spec.actions.num_values), dtype=float
    )
    values_spec = specs.Array(shape=(num_unroll_steps + 1,), dtype=float)
    actions_spec = specs.BoundedArray(
        shape=(num_unroll_steps + 1,),
        dtype=int,
        minimum=0,
        maximum=env_spec.actions.num_values,
    )

    train_target_spec = mz.utils.MooZiTrainTarget(
        observations=observations_spec,
        actions=actions_spec,
        child_visits=child_visits_spec,
        last_rewards=env_spec.rewards,
        values=values_spec,
    )

    return tree.map_structure_with_path(
        base.spec_like_to_tensor_spec, train_target_spec
    )


# def make_signature(
#     env_spec: specs.EnvironmentSpec, num_unroll_steps: int, num_stacked_frames: int
# ):
#     orig_obs_spec = env_spec.observations.observation
#     stacked_frames_shape = (num_stacked_frames,) + orig_obs_spec.shape
#     observations_spec = env_spec.observations._replace(
#         observation=orig_obs_spec.replace(shape=stacked_frames_shape)
#     )
#     last_rewards_spec = env_spec.rewards.replace(
#         shape=(num_unroll_steps + 1,) + env_spec.rewards.shape, dtype=float
#     )
#     child_visits_spec = specs.Array(
#         shape=(num_unroll_steps + 1, env_spec.actions.num_values), dtype=float
#     )
#     values_spec = specs.Array(shape=(num_unroll_steps + 1,), dtype=float)
#     actions_spec = specs.BoundedArray(
#         shape=(num_unroll_steps + 1,),
#         dtype=int,
#         minimum=0,
#         maximum=env_spec.actions.num_values,
#     )

#     train_target_spec = mz.utils.MooZiTrainTarget(
#         observations=observations_spec,
#         actions=actions_spec,
#         child_visits=child_visits_spec,
#         last_rewards=last_rewards_spec,
#         values=values_spec,
#     )

#     return tree.map_structure_with_path(
#         base.spec_like_to_tensor_spec, train_target_spec
#     )
