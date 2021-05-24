import functools
import typing

import haiku as hk
import jax
import jax.numpy as jnp

from moozi import utils


class NeuralNetwork(typing.NamedTuple):
    init: typing.Callable
    initial_inference: typing.Callable
    recurrent_inference: typing.Callable


class NeuralNetworkSpec(typing.NamedTuple):
    dim_image: int
    dim_repr: int
    dim_action: int
    # random_key: jax.random.PRNGKey


def get_network(spec: NeuralNetworkSpec):
    nn_func = functools.partial(_NeuralNetworkHaiku, spec)
    initial_inference = hk.without_apply_rng(
        hk.transform(lambda image: nn_func().initial_inference(image))
    )
    recurrent_inference = hk.without_apply_rng(
        hk.transform(lambda h, a: nn_func().recurrent_inference(h, a))
    )

    def init(random_key):
        key_1, key_2 = jax.random.split(random_key)
        params = hk.data_structures.merge(
            initial_inference.init(key_1, jnp.ones(spec.dim_image)),
            recurrent_inference.init(
                key_2, jnp.ones(spec.dim_repr), jnp.ones(spec.dim_action)
            ),
        )
        return params

    return NeuralNetwork(init, initial_inference.apply, recurrent_inference.apply)


class _NeuralNetworkHaiku(hk.Module):
    def __init__(self, spec: NeuralNetworkSpec):
        super().__init__()
        self.spec = spec

    def repr_net(self, image):
        net = hk.nets.MLP(output_sizes=[16, 16, self.spec.dim_repr], name="repr")
        return net(image)

    def pred_net(self, hidden_state):
        v_net = hk.nets.MLP(output_sizes=[16, 16, 1], name="pred_v")
        p_net = hk.nets.MLP(output_sizes=[16, 16, self.spec.dim_action], name="pred_p")
        return v_net(hidden_state), p_net(hidden_state)

    def dyna_net(self, hidden_state, action):
        state_action_repr = jnp.concatenate((hidden_state, action), axis=-1)
        transition_net = hk.nets.MLP(
            output_sizes=[16, 16, self.spec.dim_repr], name="dyna_trans"
        )
        reward_net = hk.nets.MLP(
            output_sizes=[16, 16, self.spec.dim_repr], name="dyna_reward"
        )
        return transition_net(state_action_repr), reward_net(state_action_repr)

    def initial_inference(self, image):
        hidden_state = self.repr_net(image)
        reward = 0
        value, policy_logits = self.pred_net(hidden_state)
        return utils.NetworkOutput(
            value=value,
            reward=reward,
            policy_logits=policy_logits,
            hidden_state=hidden_state,
        )

    def recurrent_inference(self, hidden_state, action):
        hidden_state, reward = self.dyna_net(hidden_state, action)
        value, policy_logits = self.pred_net(hidden_state)
        return utils.NetworkOutput(
            value=value,
            reward=reward,
            policy_logits=policy_logits,
            hidden_state=hidden_state,
        )