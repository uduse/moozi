import optax
from acme.agents import agent as acme_agent
from acme.agents import replay as acme_replay

import moozi


class RandomAgent(acme_agent.Agent):
    def __init__(self, env_spec, config, network):
        reverb_replay = acme_replay.make_reverb_prioritized_nstep_replay(
            env_spec, batch_size=config.batch_size
        )

        self._server = reverb_replay.server

        optimizer = optax.adam(config.learning_rate)
        actor = moozi.actor.RandomActor(reverb_replay.adder)
        learner = DoNothingLearner()
        super().__init__(
            actor=actor,
            learner=learner,
            min_observations=config.min_observation,
            observations_per_step=config.observations_per_step,
        )
