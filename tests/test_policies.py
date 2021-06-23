from acme.jax.utils import add_batch_dim
from acme.jax.variable_utils import VariableClient
import jax
import numpy as np
import jax.numpy as jnp
import pytest
from moozi.policies import PolicyFeed, PriorPolicy, RandomPolicy, MonteCarlo
from moozi.nerual_network import NeuralNetwork


@pytest.fixture
def policy_feed(env, num_stacked_frames, random_key) -> PolicyFeed:
    legal_actions_mask = np.zeros(4)
    legal_actions_indices = [1, 2, 3]
    legal_actions_mask[legal_actions_indices] = 1
    legal_actions_mask = jnp.array(legal_actions_mask)
    timestep = env.reset()
    frame = timestep.observation[0].observation
    stacked_frames = jnp.stack([frame.copy() for _ in range(num_stacked_frames)])

    return PolicyFeed(
        stacked_frames=stacked_frames,
        legal_actions_mask=legal_actions_mask,
        random_key=random_key,
    )


def test_random_policy_sanity(policy_feed: PolicyFeed):
    policy = RandomPolicy()
    result = policy.run(policy_feed)
    action_is_legal = policy_feed.legal_actions_mask[result.action] == 1
    assert action_is_legal


def test_prior_poliy_sanity(
    policy_feed: PolicyFeed,
    network: NeuralNetwork,
    variable_client: VariableClient,
):
    policy = PriorPolicy(
        network=network,
        variable_client=variable_client,
        epsilon=0.1,
        temperature=1,
    )
    result = policy.run(policy_feed)
    action_is_legal = policy_feed.legal_actions_mask[result.action] == 1
    assert action_is_legal


def test_monte_carlo_sanity(policy_feed, network, variable_client):
    policy = MonteCarlo(
        network=network,
        variable_client=variable_client,
        num_unroll_steps=3,
        num_simulations_per_action=20,
    )

    result = policy.run(policy_feed)
    action_is_legal = policy_feed.legal_actions_mask[result.action] == 1
    assert action_is_legal
