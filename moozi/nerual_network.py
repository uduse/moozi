import functools
import typing

import chex
import haiku as hk
import jax
import jax.numpy as jnp
from torch.nn.modules import activation

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
    pred_net_sizes: tuple = (16, 16)
    dyna_net_sizes: tuple = (16, 16)


def get_network(spec: NeuralNetworkSpec):
    # TODO: rename to build network
    hk_module = functools.partial(MLPNet, spec)
    initial_inference = hk.without_apply_rng(
        hk.transform(lambda image: hk_module().initial_inference(image))
    )
    recurrent_inference = hk.without_apply_rng(
        hk.transform(lambda h, a: hk_module().recurrent_inference(h, a))
    )

    def init(random_key):
        key_1, key_2 = jax.random.split(random_key)
        batch_size = 1
        params = hk.data_structures.merge(
            initial_inference.init(key_1, jnp.ones((batch_size, spec.dim_image))),
            recurrent_inference.init(
                key_2,
                jnp.ones((batch_size, spec.dim_repr)),
                # input actions will be scalar values and converted to one-hot ad-hoc
                jnp.ones((batch_size,)),
            ),
        )
        return params

    return NeuralNetwork(init, initial_inference.apply, recurrent_inference.apply)


class MLPNet(hk.Module):
    """
    NOTE: input tensors are assumed to have batch dimensions
    """

    def __init__(self, spec: NeuralNetworkSpec):
        super().__init__()
        self.spec = spec

    def repr_net(self, image):
        net = hk.nets.MLP(
            output_sizes=[*self.spec.repr_net_sizes, self.spec.dim_repr],
            name="repr",
            activation=jnp.tanh,
            activate_final=True,
        )
        return net(image)

    def pred_net(self, hidden_state):
        pred_trunk = hk.nets.MLP(
            output_sizes=self.spec.pred_net_sizes,
            name="pred_trunk",
            activation=jnp.tanh,
            activate_final=True,
        )
        v_branch = hk.nets.MLP(output_sizes=[1], name="pred_v", activate_final=True)
        p_branch = hk.nets.MLP(
            output_sizes=[self.spec.dim_action], name="pred_p", activate_final=True
        )

        pred_trunk_out = pred_trunk(hidden_state)
        value = jnp.squeeze(v_branch(pred_trunk_out), axis=-1)
        policy_logits = p_branch(pred_trunk_out)
        return value, policy_logits

    def dyna_net(self, hidden_state, action):
        dyna_trunk = hk.nets.MLP(
            output_sizes=self.spec.dyna_net_sizes,
            name="dyna_trunk",
            activation=jnp.tanh,
            activate_final=True,
        )
        trans_branch = hk.nets.MLP(
            output_sizes=[self.spec.dim_repr],
            name="dyna_trans",
            activation=jnp.tanh,
            activate_final=True,
        )
        reward_branch = hk.nets.MLP(
            output_sizes=[1],
            name="dyna_reward",
            activation=jnp.tanh,
            activate_final=True,
        )

        action = jax.nn.one_hot(action, num_classes=self.spec.dim_action)
        chex.assert_equal_rank([hidden_state, action])
        state_action_repr = jnp.concatenate((hidden_state, action), axis=-1)
        dyna_trunk_out = dyna_trunk(state_action_repr)
        next_hidden_states = trans_branch(dyna_trunk_out)
        next_rewards = jnp.squeeze(reward_branch(dyna_trunk_out), axis=-1)
        return next_hidden_states, next_rewards

    def initial_inference(self, image):
        chex.assert_rank(image, 2)
        hidden_state = self.repr_net(image)
        value, policy_logits = self.pred_net(hidden_state)
        reward = jnp.zeros_like(value)
        chex.assert_rank([value, reward, policy_logits, hidden_state], [1, 1, 2, 2])
        return NeuralNetworkOutput(
            value=value,
            reward=reward,
            policy_logits=policy_logits,
            hidden_state=hidden_state,
        )

    def recurrent_inference(self, hidden_state, action):
        # TODO: a batch-jit that infers K times?
        hidden_state, reward = self.dyna_net(hidden_state, action)
        value, policy_logits = self.pred_net(hidden_state)
        chex.assert_rank([value, reward, policy_logits, hidden_state], [1, 1, 2, 2])
        return NeuralNetworkOutput(
            value=value,
            reward=reward,
            policy_logits=policy_logits,
            hidden_state=hidden_state,
        )
