# %%
import typing

import acme
import acme.jax.utils
import acme.jax.variable_utils
import acme.wrappers
import chex
import dm_env
import jax
import jax.numpy as jnp
import moozi as mz
import numpy as np
import open_spiel
import optax
import reverb
import tree
from absl.testing import absltest, parameterized
from acme.adders.reverb import test_utils as acme_test_utils
from acme.adders.reverb.base import ReverbAdder
from acme.agents import agent as acme_agent
from acme.agents import replay as acme_replay
from acme.environment_loops.open_spiel_environment_loop import OpenSpielEnvironmentLoop

# %%
use_jit = True
if use_jit:
    jax.config.update("jax_disable_jit", not use_jit)

# %%
class MuZeroTrainTarget(typing.NamedTuple):
    value: float
    reward: float
    child_visits: typing.List[int]


class MuZeroAdder(ReverbAdder):
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
        

        self._writer.create_item()

# # %%
# TEST_CASES = [
#     dict(
#         testcase_name="OneStepFinalReward",
#         n_step=1,
#         additional_discount=1.0,
#         first=dm_env.restart(1),
#         steps=(
#             (0, dm_env.transition(reward=0.0, observation=2)),
#             (0, dm_env.transition(reward=0.0, observation=3)),
#             (0, dm_env.termination(reward=1.0, observation=4)),
#         ),
#         expected_transitions=(
#             (1, 0, 0.0, 1.0, 2),
#             (2, 0, 0.0, 1.0, 3),
#             (3, 0, 1.0, 0.0, 4),
#         ),
#     ),
# ]


# num_unroll_steps = 3
# discount = 1.0


# # %%
# # need: root_values, discount, rewards, child_visits
# def make_target(num_unroll_steps, td_steps):