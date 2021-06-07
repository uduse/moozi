import typing

import copy
import chex
import numpy as np
import reverb
import tree
from acme import specs as acme_specs
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

    def signature(self, env_spec):
        rewards_spec, step_discounts_spec = acme_tree_utils.broadcast_structures(
            env_spec.rewards, env_spec.discounts
        )
        rewards_spec = tree.map_structure(
            _broadcast_specs, rewards_spec, step_discounts_spec
        )
        step_discounts_spec = tree.map_structure(copy.deepcopy, step_discounts_spec)

        transition_spec = types.Transition(
            env_spec.observations,
            env_spec.actions,
            rewards_spec,
            step_discounts_spec,
            env_spec.observations,  # next_observation
            extras_spec,
        )

        return tree.map_structure_with_path(
            base.spec_like_to_tensor_spec, transition_spec
        )
        # print('hello')
        # acme_tree_utils.broadcast_structures()
