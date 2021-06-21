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


# class MooZiAgentSpec(typing.NamedTuple):
#     learner: acme.Learner
#     network: mz.nn.NeuralNetwork
#     loss_fn: typing.Callable
#     optimizer: optax.GradientTransformation
#     batch_size: int = 32
#     # learning_rate: float = 1e-3

#     # TODO: assign better values https://github.com/deepmind/acme/blob/a6b4162701542ed08b0b911ffac7c69fcb1bb3c7/acme/agents/jax/dqn/agent.py#L91
#     min_observation: int = 0
#     observations_per_step: float = 1


# class MooZiAgent(acme_agent.Agent):
#     def __init__(self, env_spec, agent_spec: MooZiAgentSpec):
#         reverb_replay = acme_replay.make_reverb_prioritized_nstep_replay(
#             env_spec, batch_size=agent_spec.batch_size, n_step=3
#         )
#         self._server = reverb_replay.server

#         actor = mz.actor.RandomActor(reverb_replay.adder)

#         learner = mz.learner.MooZiLearner(
#             network=agent_spec.network,
#             loss_fn=mz.loss.initial_inference_value_loss,
#             optimizer=agent_spec.optimizer,
#             data_iterator=reverb_replay.data_iterator,
#             random_key=jax.random.PRNGKey(0),
#         )

#         super().__init__(
#             actor=actor,
#             learner=learner,
#             min_observations=agent_spec.min_observation,
#             observations_per_step=agent_spec.observations_per_step,
#         )


# class MyAgent(acme_agent.Agent):
#     def __init__(self, env_spec, learner, actor):
#         reverb_replay = acme_replay.make_reverb_prioritized_nstep_replay(
#             env_spec, batch_size=agent_spec.batch_size, n_step=3
#         )
#         self._server = reverb_replay.server

#         actor = mz.actor.RandomActor(reverb_replay.adder)

#         learner = mz.learner.MooZiLearner(
#             network=agent_spec.network,
#             loss_fn=mz.loss.initial_inference_value_loss,
#             optimizer=agent_spec.optimizer,
#             data_iterator=reverb_replay.data_iterator,
#             random_key=jax.random.PRNGKey(0),
#         )

#         super().__init__(
#             actor=actor,
#             learner=learner,
#             min_observations=agent_spec.min_observation,
#             observations_per_step=agent_spec.observations_per_step,
#         )
