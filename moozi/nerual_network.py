import functools
import typing

import chex
import haiku as hk
import jax
import jax.numpy as jnp

import moozi as mz


@chex.dataclass(frozen=True)
class NeuralNetworkOutput:
    value: float
    reward: float
    # policy_logits: typing.Dict[mz.Action, float]
    policy_logits: chex.Array
    hidden_state: chex.Array


class NeuralNetwork(typing.NamedTuple):
    init: typing.Callable
    initial_inference: typing.Callable[..., NeuralNetworkOutput]
    recurrent_inference: typing.Callable[..., NeuralNetworkOutput]


class NeuralNetworkSpec(typing.NamedTuple):
    dim_image: int
    dim_repr: int
    dim_action: int
    repr_net_sizes: tuple = (16, 16)
    pred_net_sizes: list = (16, 16)
    dyna_net_sizes: list = (16, 16)


# def get_mini_nn_spec(dim_image):
#     return NeuralNetworkSpec()


def get_network(spec: NeuralNetworkSpec):
    # TODO: rename to build network
    hk_module = functools.partial(_NeuralNetworkHaiku, spec)
    initial_inference = hk.without_apply_rng(
        hk.transform(lambda image: hk_module().initial_inference(image))
    )
    recurrent_inference = hk.without_apply_rng(
        hk.transform(lambda h, a: hk_module().recurrent_inference(h, a))
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
        net = hk.nets.MLP(
            output_sizes=[*self.spec.repr_net_sizes, self.spec.dim_repr], name="repr"
        )
        # NOTE: output is relu-ed, maybe it shouldn't be
        return net(image)

    def pred_net(self, hidden_state):
        pred_trunk = hk.nets.MLP(
            output_sizes=self.spec.pred_net_sizes, name="pred_trunk"
        )
        v_branch = hk.Linear(output_size=1, name="pred_v")
        p_branch = hk.Linear(output_size=self.spec.dim_action, name="pred_p")

        pred_trunk_out = pred_trunk(hidden_state)
        return v_branch(pred_trunk_out), p_branch(pred_trunk_out)

    def dyna_net(self, hidden_state, action):
        dyna_trunk = hk.nets.MLP(
            output_sizes=self.spec.dyna_net_sizes, name="dyna_trunk"
        )
        trans_branch = hk.Linear(output_size=self.spec.dim_repr, name="dyna_trans")
        reward_branch = hk.Linear(output_size=1, name="dyna_reward")

        state_action_repr = jnp.concatenate((hidden_state, action), axis=-1)
        dyna_trunk_out = dyna_trunk(state_action_repr)
        return trans_branch(dyna_trunk_out), reward_branch(dyna_trunk_out)

    def initial_inference(self, image):
        hidden_state = self.repr_net(image)
        reward = 0
        value, policy_logits = self.pred_net(hidden_state)
        return NeuralNetworkOutput(
            value=value,
            reward=reward,
            policy_logits=policy_logits,
            hidden_state=hidden_state,
        )

    def recurrent_inference(self, hidden_state, action):
        hidden_state, reward = self.dyna_net(hidden_state, action)
        value, policy_logits = self.pred_net(hidden_state)
        return NeuralNetworkOutput(
            value=value,
            reward=reward,
            policy_logits=policy_logits,
            hidden_state=hidden_state,
        )
