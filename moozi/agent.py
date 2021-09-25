import typing
import acme
import jax
import optax
from acme.agents import agent as acme_agent
from acme.agents import replay as acme_replay

import moozi as mz


class RandomAgent(acme_agent.Agent):
    def __init__(
        self, env_spec, batch_size=32, min_observation=100, observations_per_step=1
    ):
        reverb_replay = acme_replay.make_reverb_prioritized_nstep_replay(
            env_spec, batch_size=batch_size
        )
        self._server = reverb_replay.server

        actor = mz.actor.RandomActor(reverb_replay.adder)
        learner = mz.learner.NoOpLearner()
        super().__init__(
            actor=actor,
            learner=learner,
            min_observations=min_observation,
            observations_per_step=observations_per_step,
        )
