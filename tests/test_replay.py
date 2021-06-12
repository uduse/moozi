# from absl.testing import absltest
import typing

import chex
import dm_env
import reverb
import moozi as mz
import numpy as np
import tree
from absl.testing import parameterized
from acme import specs as acme_specs
from acme.adders.reverb import test_utils as acme_test_utils
from moozi.adder import MooZiAdder
from moozi.types import Action
from moozi.utils import MooZiObservation, MooZiTrainTarget

TEST_CASES = [
    dict(
        testcase_name="BoardGameN1",
        num_unroll_steps=0,
        num_stacked_images=1,
        num_td_steps=0,
        discount=1,
        observations=[
            MooZiObservation(
                dm_env.restart(0), root_value=0, child_visits=[0.1, 0.3, 0.6]
            ),
            MooZiObservation(
                dm_env.transition(reward=0, observation=1, discount=0.5),
                root_value=10,
                child_visits=[3, 4, 5],
            ),
        ],
        targets=[MooZiTrainTarget()],
    ),
]


def _numeric_to_spec(x: typing.Union[float, int, np.ndarray]):
    if isinstance(x, np.ndarray):
        return acme_specs.Array(shape=x.shape, dtype=x.dtype)
    elif isinstance(x, (float, int)):
        return acme_specs.Array(shape=(), dtype=type(x))
    else:
        raise ValueError(f"Unsupported numeric: {type(x)}")




# class MooZiAdderTest(acme_test_utils.AdderTestMixin, parameterized.TestCase):
#     @parameterized.named_parameters(*TEST_CASES)
#     def test_adder(
#         self,
#         num_unroll_steps,
#         num_stacked_images,
#         num_td_steps,
#         discount,
#         observations,
#         targets,
#     ):
#         adder = MooZiAdder(
#             self.client,
#             num_unroll_steps=num_unroll_steps,
#             num_stacked_images=num_stacked_images,
#             num_td_steps=num_td_steps,
#             discount=discount,
#         )
