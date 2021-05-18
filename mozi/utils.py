import typing
import random


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
            raise ValueError("{} elements could not be sampled from size {}".format(
                num_samples, len(self._data)))
        return random.sample(self._data, num_samples)

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)


Action = typing.NewType('Action', int)


class NetworkOutput(typing.NamedTuple):
    value: float
    reward: float
    policy_logits: typing.Dict[Action, float]
    hidden_state: typing.List[float]


class Network(object):
    def __init__(self, params):
        self.params = params

    def initial_inference(self, input_) -> NetworkOutput:
        return

    def recurrent_inference(self, input_) -> NetworkOutput:
        return
