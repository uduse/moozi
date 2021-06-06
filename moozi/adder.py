import typing

import chex
import numpy as np
import reverb
import tree
from acme.adders import reverb as acme_reverb
from acme.adders.reverb import base
from acme.adders.reverb import utils as acme_reverb_utils


class MuZeroTrainTarget(typing.NamedTuple):
    observation: chex.ArrayDevice
    value: float
    reward: float
    child_visits: typing.List[int]

# class Step(NamedTuple):
#   """Step class used internally for reverb adders."""
#   observation: types.NestedArray
#   action: types.NestedArray
#   reward: types.NestedArray
#   discount: types.NestedArray
#   start_of_episode: StartOfEpisodeType
#   extras: types.NestedArray = ()

# need: root_values, discount, rewards, child_visits


class MuZeroAdder(acme_reverb.ReverbAdder):
    def __init__(
        self,
        client: reverb.Client,
        num_unroll_steps: int,
        # td_steps: int,
        discount: float,
    ):
        self._num_unroll_steps = num_unroll_steps
        self._discount = tree.map_structure(np.float32, discount)

        # according to the pseudocode, 500 is roughly enough for board games
        max_sequence_length = 500
        # use full monte-carlo return for board games
        self._td_steps = max_sequence_length
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
        self._writer.create_item(acme_reverb.DEFAULT_PRIORITY_TABLE, 1, trajectory)
