import numpy as np

import moozi
from moozi import *
from moozi.utils import NetworkOutput, Network

DIM_ACTIONS = 10


class TestNetwork(Network):
    def initial_inference(self, features) -> NetworkOutput:
        hidden_state = features
        value = np.mean(hidden_state)
        reward = np.std(hidden_state)
        policy_logits = {
            Action(a): hidden_state[a] for a in range(DIM_ACTIONS)
        }
        return NetworkOutput(value, reward, policy_logits, hidden_state)

    def recurrent_inference(self, hidden_state, action) -> NetworkOutput:
        return


def test_mcts():
    from moozi import mcts
    config = Config()
    root = Node(1)
    network = TestNetwork()
    obs = list(range(5))
    network_output = network.initial_inference(config, obs)
    action_history = ActionHistory([], DIM_ACTIONS)
    mcts.run_mcts(config, root, action_history, network)
