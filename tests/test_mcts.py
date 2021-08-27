# import numpy as np

# import moozi
# from moozi import *
# from moozi.utils import NetworkOutput, Network

# DIM_ACTIONS = 10


# class TestNetwork(Network):
#     def initial_inference(self, features) -> NetworkOutput:
#         hidden_state = features
#         value = np.mean(hidden_state)
#         reward = np.std(hidden_state)
#         policy_logits = {
#             Action(a): hidden_state[a] for a in range(DIM_ACTIONS)
#         }
#         return NetworkOutput(value, reward, policy_logits, hidden_state)

#     def recurrent_inference(self, hidden_state, action) -> NetworkOutput:
#         value = np.mean(hidden_state)
#         reward = np.std(hidden_state)
#         policy_logits = {
#             Action(a): hidden_state[a] for a in range(DIM_ACTIONS)
#         }
#         return NetworkOutput(value, reward, policy_logits, hidden_state)


# def test_mcts():
#     config = Config(DIM_ACTIONS, 10, 0.5, 0.1, 25, 1, 1, 1, 1e-3, 1e-4, None)
#     root = Node(1)
#     network = TestNetwork()
#     legal_actions = list(map(Action, range(DIM_ACTIONS)))
#     obs = list(range(DIM_ACTIONS))
#     network_output = network.initial_inference(obs)
#     mcts.expand_node(root, Player(0), legal_actions, network_output)
#     action_history = ActionHistory([], DIM_ACTIONS)
#     mcts.run_mcts(config, root, action_history, network)

def test_mcts():
    