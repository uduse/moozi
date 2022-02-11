from dataclasses import dataclass
import functools
from typing import Callable, NamedTuple, Tuple
from acme.jax.utils import add_batch_dim, squeeze_batch_dim

import chex
import haiku as hk
import jax
import jax.numpy as jnp


class NNOutput(NamedTuple):
    value: jnp.ndarray
    reward: jnp.ndarray
    policy_logits: jnp.ndarray
    hidden_state: jnp.ndarray


class RootInferenceFeatures(NamedTuple):
    stacked_frames: jnp.ndarray
    player: jnp.ndarray


class TransitionInferenceFeatures(NamedTuple):
    hidden_state: jnp.ndarray
    action: jnp.ndarray


class NeuralNetworkSpec(NamedTuple):
    stacked_frames_shape: tuple
    dim_repr: int
    dim_action: int

class NeuralNetwork(NamedTuple):
    init_network: Callable
    root_inference: Callable[..., Tuple[NNOutput, ...]]
    trans_inference: Callable[..., Tuple[NNOutput, ...]]
    root_inference_unbatched: Callable[..., Tuple[NNOutput, ...]]
    trans_inference_unbatched: Callable[..., Tuple[NNOutput, ...]]



def get_network(spec: NeuralNetworkSpec):
    # TODO: rename to build network
    hk_module = functools.partial(MLPNet, spec)
    initial_inference = hk.without_apply_rng(
        hk.transform(
            lambda init_inf_features: hk_module().initial_inference(init_inf_features)
        )
    )
    recurrent_inference = hk.without_apply_rng(
        hk.transform(
            lambda recurr_inf_features: hk_module().recurrent_inference(
                recurr_inf_features
            )
        )
    )

    def init(random_key):
        key_1, key_2 = jax.random.split(random_key)
        batch_size = 1
        params = hk.data_structures.merge(
            initial_inference.init(
                key_1,
                RootInferenceFeatures(
                    stacked_frames=jnp.ones((batch_size,) + spec.stacked_frames_shape),
                    player=jnp.array(0),
                ),
            ),
            recurrent_inference.init(
                key_2,
                TransitionInferenceFeatures(
                    hidden_state=jnp.ones((batch_size, spec.dim_repr)),
                    action=jnp.ones((batch_size,)),
                ),
            ),
        )
        return params

    def _initial_inference_unbatched(params, init_inf_features):
        return squeeze_batch_dim(
            initial_inference.apply(params, add_batch_dim(init_inf_features))
        )

    def _recurrent_inference_unbatched(params, recurr_inf_features):
        return squeeze_batch_dim(
            recurrent_inference.apply(params, add_batch_dim(recurr_inf_features))
        )

    return NeuralNetwork(
        init,
        initial_inference.apply,
        recurrent_inference.apply,
        _initial_inference_unbatched,
        _recurrent_inference_unbatched,
    )


# deprecated
# class NeuralNetworkSpec(NamedTuple):
#     stacked_frames_shape: tuple
#     dim_repr: int
#     dim_action: int
#     repr_net_sizes: tuple = (16, 16)
#     pred_net_sizes: tuple = (16, 16)
#     dyna_net_sizes: tuple = (16, 16)

class MLPNet(hk.Module):
    """
    NOTE: input tensors are assumed to have batch dimensions
    """

    def __init__(self, spec: NeuralNetworkSpec):
        super().__init__()
        self.spec = spec

    def repr_net(self, flattened_frames):
        net = hk.nets.MLP(
            output_sizes=[*self.spec.repr_net_sizes, self.spec.dim_repr],
            name="repr",
            activation=jnp.tanh,
            activate_final=True,
        )
        return net(flattened_frames)

    def pred_net(self, hidden_state):
        pred_trunk = hk.nets.MLP(
            output_sizes=self.spec.pred_net_sizes,
            name="pred_trunk",
            activation=jnp.tanh,
            activate_final=True,
        )
        v_branch = hk.Linear(output_size=1, name="pred_v")
        p_branch = hk.Linear(output_size=self.spec.dim_action, name="pred_p")

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

        action_one_hot = jax.nn.one_hot(action, num_classes=self.spec.dim_action)
        chex.assert_equal_rank([hidden_state, action_one_hot])
        state_action_repr = jnp.concatenate((hidden_state, action_one_hot), axis=-1)
        dyna_trunk_out = dyna_trunk(state_action_repr)
        next_hidden_states = trans_branch(dyna_trunk_out)
        next_rewards = jnp.squeeze(reward_branch(dyna_trunk_out), axis=-1)
        return next_hidden_states, next_rewards

    def initial_inference(self, init_inf_features: RootInferenceFeatures):
        chex.assert_rank(init_inf_features.stacked_frames, 3)
        flattened_stacked_frames = hk.Flatten()(init_inf_features.stacked_frames)
        hidden_state = self.repr_net(flattened_stacked_frames)
        value, policy_logits = self.pred_net(hidden_state)
        reward = jnp.zeros_like(value)
        chex.assert_rank([value, reward, policy_logits, hidden_state], [1, 1, 2, 2])
        return NNOutput(
            value=value,
            reward=reward,
            policy_logits=policy_logits,
            hidden_state=hidden_state,
        )

    def recurrent_inference(self, recurr_inf_features: TransitionInferenceFeatures):
        # TODO: a batch-jit that infers K times?
        next_hidden_state, reward = self.dyna_net(
            recurr_inf_features.hidden_state, recurr_inf_features.action
        )
        value, policy_logits = self.pred_net(next_hidden_state)
        chex.assert_rank(
            [value, reward, policy_logits, next_hidden_state], [1, 1, 2, 2]
        )
        return NNOutput(
            value=value,
            reward=reward,
            policy_logits=policy_logits,
            hidden_state=next_hidden_state,
        )


class ConvolutionBlock(hk.Module):
    def __call__(self, x):
        x = hk.Conv2D(256, (3, 3), padding="same")(x)
        x = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.9)(
            x, is_training=True
        )
        x = jax.nn.relu(x)
        return x


class ResidueBlock(hk.Module):
    def __call__(self, x):
        orig_x = x
        x = hk.Conv2D(output_channels=256, kernel_shape=(3, 3), padding="same")(x)
        x = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.9)(
            x, is_training=True
        )
        x = jax.nn.relu(x)
        x = hk.Conv2D(output_channels=256, kernel_shape=(3, 3), padding="same")(x)
        x = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.9)(
            x, is_training=True
        )
        x = x + orig_x
        x = jax.nn.relu(x)
        return x


@dataclass
class ResidueTower(hk.Module):
    num_blocks: int

    def __call__(self, x):
        for _ in range(self.num_blocks):
            x = ResidueBlock()(x)
        return x


# @dataclass
# class HeadBlock(hk.Module):
#     num_outputs: int

#     def __call__(self, x):
#         x = hk.Conv2D(output_channels=2, kernel_shape=(1, 1), padding="same")(x)
#         x = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.9)(
#             x, is_training=True
#         )
#         x = jax.nn.relu(x)


class MuZeroNet(hk.Module):
    def __init__(self, spec: NeuralNetworkSpec):
        super().__init__()
        self.spec = spec

    def repr_net(self, stacked_frames):
        x = ConvolutionBlock()(stacked_frames)
        x = ResidueTower(num_blocks=16)(x)
        x = hk.Conv2D(output_channels=512, kernel_shape=(3, 3), padding="same")(x)

    def pred_net(self, hidden_state):
        pred_trunk = hk.nets.MLP(
            output_sizes=self.spec.pred_net_sizes,
            name="pred_trunk",
            activation=jnp.tanh,
            activate_final=True,
        )
        v_branch = hk.Linear(output_size=1, name="pred_v")
        p_branch = hk.Linear(output_size=self.spec.dim_action, name="pred_p")

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

        action_one_hot = jax.nn.one_hot(action, num_classes=self.spec.dim_action)
        chex.assert_equal_rank([hidden_state, action_one_hot])
        state_action_repr = jnp.concatenate((hidden_state, action_one_hot), axis=-1)
        dyna_trunk_out = dyna_trunk(state_action_repr)
        next_hidden_states = trans_branch(dyna_trunk_out)
        next_rewards = jnp.squeeze(reward_branch(dyna_trunk_out), axis=-1)
        return next_hidden_states, next_rewards

    def initial_inference(self, init_inf_features: RootInferenceFeatures):
        chex.assert_rank(init_inf_features.stacked_frames, 3)
        hidden_state = self.repr_net(init_inf_features.stacked_frames)
        value, policy_logits = self.pred_net(hidden_state)
        reward = jnp.zeros_like(value)
        chex.assert_rank([value, reward, policy_logits, hidden_state], [1, 1, 2, 2])
        return NNOutput(
            value=value,
            reward=reward,
            policy_logits=policy_logits,
            hidden_state=hidden_state,
        )

    def recurrent_inference(self, recurr_inf_features: TransitionInferenceFeatures):
        # TODO: a batch-jit that infers K times?
        next_hidden_state, reward = self.dyna_net(
            recurr_inf_features.hidden_state, recurr_inf_features.action
        )
        value, policy_logits = self.pred_net(next_hidden_state)
        chex.assert_rank(
            [value, reward, policy_logits, next_hidden_state], [1, 1, 2, 2]
        )
        return NNOutput(
            value=value,
            reward=reward,
            policy_logits=policy_logits,
            hidden_state=next_hidden_state,
        )
