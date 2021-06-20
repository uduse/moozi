import dm_env
from nptyping import NDArray
import numpy as np
from acme.core import Actor
from acme.wrappers.open_spiel_wrapper import OLT


class RandomActor(Actor):
    def __init__(self, adder):
        self._adder = adder

    def select_action(self, observation: OLT) -> int:
        legals = np.array(np.nonzero(observation.legal_actions), dtype=np.int32)
        return np.random.choice(legals[0])

    def observe_first(self, timestep: dm_env.TimeStep):
        self._adder.add_first(timestep)

    def observe(self, action: NDArray[np.int32], next_timestep: dm_env.TimeStep):
        self._adder.add(action, next_timestep)

    def update(self, wait: bool = False):
        pass
