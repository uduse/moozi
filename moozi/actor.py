import acme
import acme.jax.utils as acme_utils
import acme.wrappers.open_spiel_wrapper
import acme.jax.variable_utils
import chex
import dm_env
import jax
import jax.numpy as jnp
import numpy as np
import rlax

import moozi as mz


class RandomActor(acme.core.Actor):
    def __init__(self, adder):
        self._adder = adder

    def select_action(self, observation: acme.wrappers.open_spiel_wrapper.OLT) -> int:
        legals = np.array(np.nonzero(observation.legal_actions), dtype=np.int32)
        return np.random.choice(legals[0])

    def observe_first(self, timestep: dm_env.TimeStep):
        self._adder.add_first(timestep)

    def observe(self, action: acme.types.NestedArray, next_timestep: dm_env.TimeStep):
        self._adder.add(action, next_timestep)

    def update(self, wait: bool = False):
        pass


class PriorPolicyActor(acme.core.Actor):
    def __init__(
        self,
        network: mz.nn.NeuralNetwork,
        adder,
        variable_client: acme.jax.variable_utils.VariableClient,
        random_key,
        epsilon=0.1,
        temperature=1,
    ):
        def _policy_fn(
            params, observation: acme.wrappers.open_spiel_wrapper.OLT, random_key
        ):
            # TODO: add batched dim here?
            # https://github.com/deepmind/acme/blob/926b17ad116578801a0fbbe73c4ddc276a28e23e/acme/agents/jax/actors.py#L76
            network_output = network.initial_inference(params, observation.observation)
            action_logits = network_output.policy_logits
            chex.assert_rank(action_logits, 1)
            action_logits -= observation.legal_actions * jnp.inf
            _sampler = rlax.epsilon_softmax(epsilon, temperature).sample
            action = _sampler(random_key, action_logits)
            return action

        self._random_key = random_key
        self._policy_fn = jax.jit(jax.vmap(_policy_fn, in_axes=[None, 0, None]))
        self._adder = adder
        self._client = variable_client

    def select_action(self, observation: acme.wrappers.open_spiel_wrapper.OLT) -> int:
        self._random_key, new_key = jax.random.split(self._random_key)
        result = self._policy_fn(self._client.params, observation, new_key)
        return acme_utils.to_numpy(result)

    def observe_first(self, timestep: dm_env.TimeStep):
        self._adder.add_first(timestep)

    def observe(self, action: acme.types.NestedArray, next_timestep: dm_env.TimeStep):
        self._adder.add(action, next_timestep)

    def update(self, wait: bool = False):
        pass
