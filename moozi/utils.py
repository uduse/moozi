import acme.jax.variable_utils
import typing
import random


Action = typing.NewType("Action", int)


class ReplayBuffer(object):
    def __init__(self, capacity):
        super().__init__()
        self._capacity = capacity
        self._data = []
        self._next_entry_index = 0

    def add(self, element):
        if len(self._data) < self._capacity:
            self._data.append(element)
        else:
            self._data[self._next_entry_index] = element
            self._next_entry_index += 1
            self._next_entry_index %= self._capacity

    def extend(self, elements):
        for ele in elements:
            self.add(ele)

    def sample(self, num_samples):
        if len(self._data) < num_samples:
            raise ValueError(
                "{} elements could not be sampled from size {}".format(
                    num_samples, len(self._data)
                )
            )
        return random.sample(self._data, num_samples)

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)


def print_traj_in_env(env):
    timestep = env.reset()
    while True:
        print(env.environment.environment.get_state.observation_string(0))
        actions = env.environment.get_state.legal_actions()
        action = random.choice(actions)
        print("a:", action, "\n")
        timestep = env.step([action])
        if timestep.last():
            print(env.environment.environment.get_state.observation_string(0))
            print("reward:", timestep.reward)
            break
