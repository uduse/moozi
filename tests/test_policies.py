import chex
import copy
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import rlax
from acme.jax.utils import add_batch_dim
from acme.jax.variable_utils import VariableClient
from moozi.nn import NeuralNetwork
from moozi.core import PolicyFeed
# , make_single_roll_monte_carlo_fn, policy



# def test_random_policy_sanity(policy_feed: PolicyFeed):
#     policy = RandomPolicy()
#     result = policy.run(policy_feed)
#     action_is_legal = policy_feed.legal_actions_mask[result.action] == 1
#     assert action_is_legal


# def test_prior_poliy_sanity(
#     policy_feed: PolicyFeed,
#     network: NeuralNetwork,
#     variable_client: VariableClient,
# ):
#     policy = PriorPolicy(
#         network=network,
#         variable_client=variable_client,
#         epsilon=0.1,
#         temperature=1,
#     )
#     result = policy.run(policy_feed)
#     action_is_legal = policy_feed.legal_actions_mask[result.action_probs] == 1
#     assert action_is_legal


def test_single_roll_monte_carlo(
    policy_feed: PolicyFeed, network: NeuralNetwork, params
):
    single_roll_monte_carlo_fn = make_single_roll_monte_carlo_fn(
        network=network, num_unroll_steps=3
    )
    policy_result = single_roll_monte_carlo_fn(params, policy_feed)
    policy_epsilon = 0.1
    action_probs = rlax.epsilon_greedy(policy_epsilon).probs(
        policy_result.actions_reward_sum
    )

    action = rlax.categorical_sample(policy_feed.random_key, action_probs)
    assert action in range(policy_feed.legal_actions_mask.shape[0])


def test_mcts_backpropagate(policy_feed: PolicyFeed, network: NeuralNetwork, params):
    nn_output = network.root_inference_unbatched(params, policy_feed.stacked_frames)
    root = mcts.make_root_node(nn_output.hidden_state)
    child = copy.deepcopy(root)._replace(parent=root)
    child_child = copy.deepcopy(child)._replace(parent=child)
    mcts.backpropagate(child_child, value=1.0, discount=0.99)


# def test_mcts(policy_feed, network: NeuralNetwork, params):
#     mcts = MonteCarloTreeSearch(network, num_simulations=10)
#     mcts_result = mcts(params, policy_feed)
#     print(mcts_result)
