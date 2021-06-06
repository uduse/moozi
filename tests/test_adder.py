# from absl.testing import absltest
import typing

import chex
import dm_env
from absl.testing import parameterized
from acme.adders.reverb import test_utils as acme_test_utils
from moozi.adder import MuZeroAdder
from moozi.types import Action


class Transition(typing.NamedTuple):
    observation: chex.ArrayDevice
    action: chex.ArrayDevice
    reward: chex.ArrayDevice
    discount: chex.ArrayDevice
    next_observation: chex.ArrayDevice
    extras: chex.ArrayDevice = ()


TEST_CASES = [
    dict(
        testcase_name="BoardGameN1",
        num_unroll_steps=0,
        additional_discount=1.0,
        first=dm_env.restart(0),
        steps=[(Action(0), dm_env.termination(reward=1, observation=1))],
        expected_transitions=[
            Transition(
                observation=0, action=0, reward=1, discount=1, next_observation=1
            )
        ],
    ),
]


class MuZeroAdderTest(acme_test_utils.AdderTestMixin, parameterized.TestCase):
    @parameterized.named_parameters(*TEST_CASES)
    def test_adder(
        self, num_unroll_steps, additional_discount, first, steps, expected_transitions
    ):
        adder = MuZeroAdder(self.client, num_unroll_steps, additional_discount)