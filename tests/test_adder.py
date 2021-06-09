# from absl.testing import absltest
import typing

import chex
import dm_env
import moozi as mz
import numpy as np
import tree
from absl.testing import parameterized
from acme import specs as acme_specs
from acme.adders.reverb import test_utils as acme_test_utils
from moozi.adder import MooZiAdder
from moozi.types import Action
from moozi.utils import MooZiObservation, MooZiTrainTarget


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
            MooZiObservation(dm_env.restart(0), root_value=0, child_visits=[0.1, 0.3, 0.6]),
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
    raise ValueError(f'Unsupported numeric: {type(x)}')

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

        has_extras = len(steps[0]) == 3
        env_spec = tree.map_structure(
            _numeric_to_spec,
            specs.EnvironmentSpec(
                observations=steps[0][1].observation,
                actions=steps[0][0],
                rewards=steps[0][1].reward,
                discounts=steps[0][1].discount))

        if has_extras:
            extras_spec = tree.map_structure(_numeric_to_spec, steps[0][2])
        else:
            extras_spec = ()
        signature = adder.signature(env_spec, extras_spec=extras_spec)

        for episode_id in range(repeat_episode_times):
            # Add all the data up to the final step.
            adder.add_first(first)
            for step in steps[:-1]:
            action, ts = step[0], step[1]

            if has_extras:
                extras = step[2]
            else:
                extras = ()

            adder.add(action, next_timestep=ts, extras=extras)

            # Add the final step.
            adder.add(*steps[-1])

        # Ending the episode should close the writer. No new writer should yet have
        # been created as it is constructed lazily.
        if break_end_of_episode:
            self.assertEqual(self.client.writer.num_episodes, repeat_episode_times)

        # Make sure our expected and observed data match.
        observed_items = [p[2] for p in self.client.writer.priorities]

        # Check matching number of items.
        self.assertEqual(len(expected_items), len(observed_items))

        # Check items are matching according to numpy's almost_equal.
        for expected_item, observed_item in zip(expected_items, observed_items):
            if stack_sequence_fields:
            expected_item = tree_utils.stack_sequence_fields(expected_item)

            # Set check_types=False because we check them below.
            tree.map_structure(
                np.testing.assert_array_almost_equal,
                expected_item,
                tuple(observed_item),
                check_types=False)

        # Make sure the signature matches was is being written by Reverb.
        def _check_signature(spec: tf.TensorSpec, value: np.ndarray):
            self.assertTrue(spec.is_compatible_with(tf.convert_to_tensor(value)))

        # Check the last transition's signature.
        tree.map_structure(_check_signature, signature, observed_items[-1])
