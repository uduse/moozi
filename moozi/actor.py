import acme
import dm_env
import numpy as np


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
