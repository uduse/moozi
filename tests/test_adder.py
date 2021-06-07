# from absl.testing import absltest
import typing

import chex
import dm_env
from absl.testing import parameterized
from acme.adders.reverb import test_utils as acme_test_utils
import moozi as mz
from moozi.utils import MooZiObservation, MooZiTrainTarget
from moozi.adder import MooZiAdder
from moozi.types import Action
import tree


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
        num_stacked_images=1,
        num_td_steps=0,
        discount=1,
        observations=[
            MooZiObservation(dm_env.restart(0), root_value=0, child_visits=[1, 2, 3]),
            MooZiObservation(
                dm_env.transition(reward=0, observation=1, discount=0.5),
                root_value=10,
                child_visits=[3, 4, 5],
            ),
        ],
        targets=[MooZiTrainTarget()],
    ),
]


class MooZiAdderTest(acme_test_utils.AdderTestMixin, parameterized.TestCase):
    @parameterized.named_parameters(*TEST_CASES)
    def test_adder(
        self,
        num_unroll_steps,
        num_stacked_images,
        num_td_steps,
        discount,
        observations,
        targets,
    ):
        adder = MooZiAdder(
            self.client,
            num_unroll_steps=num_unroll_steps,
            num_stacked_images=num_stacked_images,
            num_td_steps=num_td_steps,
            discount=discount,
        )
        env_spec = tree.map_structure(
            _numeric_to_spec,
            specs.EnvironmentSpec(
                observations=steps[0][1].observation,
                actions=steps[0][0],
                rewards=steps[0][1].reward,
                discounts=steps[0][1].discount,
            ),
        )
