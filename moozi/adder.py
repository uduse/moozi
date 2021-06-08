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


class MooZiAdder(acme_reverb.ReverbAdder):
    def __init__(
        self,
        client: reverb.Client,
        num_unroll_steps: int = 5,
        num_stacked_images: int = 1,
        num_td_steps: int = 1000,
        discount: float = 1,
    ):

        self._client = client
        self._num_unroll_steps = num_unroll_steps
        self._num_td_steps = num_td_steps
        self._num_stacked_images = num_stacked_images
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
        self._writer.create_item(acme_reverb.DEFAULT_PRIORITY_TABLE, 1, trajectory)

    def signature(self, env_spec: acme_specs.EnvironmentSpec):
        orig_obs_spec = env_spec.observations.observation
        stacked_frames_shape = (self._num_stacked_images,) + orig_obs_spec.shape
        observations_spec = env_spec.observations._replace(
            observation=orig_obs_spec.replace(shape=stacked_frames_shape)
        )
        rewards_spec = tree.map_structure(_broadcast_specs, env_spec.rewards)
        child_visits_spec = acme_specs.Array(
            shape=(env_spec.actions.num_values,), dtype=int
        )

        transition_spec = mz.utils.MooZiTrainTarget(
            observations=observations_spec,
            actions=env_spec.actions,
            child_visits=child_visits_spec,
            last_rewards=rewards_spec,
            values=(),
        )

        return tree.map_structure_with_path(
            base.spec_like_to_tensor_spec, transition_spec
        )


def _broadcast_specs(*args: acme_specs.Array) -> acme_specs.Array:
    bc_info = np.broadcast(*tuple(a.generate_value() for a in args))
    dtype = np.result_type(*tuple(a.dtype for a in args))
    return acme_specs.Array(shape=bc_info.shape, dtype=dtype)
