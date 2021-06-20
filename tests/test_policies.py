import jax
import numpy as np
import jax.numpy as jnp
import pytest
from moozi.policies.policy import PolicyFeed, PriorPolicy, RandomPolicy
from moozi.nerual_network import NeuralNetwork


@pytest.fixture
def policy_feed(params, env) -> PolicyFeed:
    legal_actions_mask = np.zeros(10)
    legal_actions_indices = [1, 2, 3]
    legal_actions_mask[legal_actions_indices] = 1
    legal_actions_mask = jnp.array(legal_actions_mask)
    random_key = jax.random.PRNGKey(0)
    timestep = env.reset()
    frame = timestep.observation[0].observation
    return PolicyFeed(
        params=params,
        stacked_frames=frame,
        legal_actions_mask=legal_actions_mask,
        random_key=random_key,
    )


def test_random_policy_sanity(policy_feed: PolicyFeed):
    policy = RandomPolicy()
    result = policy.run(policy_feed)
    action_is_legal = policy_feed.legal_actions_mask[result.action] == 1
    assert action_is_legal


def test_prior_poliy_sanity(policy_feed: PolicyFeed, network: NeuralNetwork):
    policy = PriorPolicy(network, epsilon=0.1, temperature=1)
    result = policy.run(policy_feed)
    action_is_legal = policy_feed.legal_actions_mask[result.action] == 1
    assert action_is_legal
