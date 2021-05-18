import haiku as hk
import jax
from jax import numpy as jnp
import pyspiel

from ..utils import *

game = pyspiel.load_game('tic_tac_toe')
key = jax.random.PRNGKey(0)
dim_image = game.observation_tensor_size()
dim_repr = 4
dim_actions = game.num_distinct_actions()
all_actions = list(range(dim_actions))


def _pred_net(hidden_state):
    v_net = hk.nets.MLP(output_sizes=[16, 16, 1])
    p_net = hk.nets.MLP(output_sizes=[16, 16, dim_actions])
    return v_net(hidden_state), p_net(hidden_state)


pred_net = hk.without_apply_rng(hk.transform(_pred_net))
key, new_key = jax.random.split(key)
pred_params = pred_net.init(new_key, jnp.ones(dim_repr))


def _dyna_net(hidden_state, action):
    state_action_repr = jnp.concatenate((hidden_state, action), axis=-1)
    transition_net = hk.nets.MLP(output_sizes=[16, 16, dim_repr])
    reward_net = hk.nets.MLP(output_sizes=[16, 16, dim_repr])
    return transition_net(state_action_repr)


dyna_net = hk.without_apply_rng(hk.transform(_dyna_net))
key, new_key = jax.random.split(key)
dyna_params = dyna_net.init(key, jnp.ones(dim_repr), jnp.ones(dim_actions))


def _repr_net(image):
    net = hk.nets.MLP(output_sizes=[16, 16, dim_repr])
    return net(image)


repr_net = hk.without_apply_rng(hk.transform(_repr_net))
key, new_key = jax.random.split(key)
repr_params = repr_net.init(key, jnp.ones(dim_board))


def initial_inference(repr_params, pred_params, image):
    hidden_state = repr_net.apply(image)
    reward = 0
    value, policy_logits = pred_net.apply(hidden_state)
    return NetworkOutput(
        value=value,
        reward=reward,
        policy_logits=policy_logits,
        hidden_state=hidden_state
    )


def recurrent_inference(dyna_params, pred_params, hidden_state, action):
    dyna_net.apply(hidden_state, action)
    value, policy_logits = pred_net.apply(hidden_state)
    return NetworkOutput(
        value=value,
        reward=reward,
        policy_logits=policy_logits,
        hidden_state=hidden_state
    )
