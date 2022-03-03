import jax
import jax.numpy as jnp
import chex
from dataclasses import dataclass
import haiku as hk

from moozi.nn import (
    NNOutput,
    NNSpec,
    RootFeatures,
    TransitionFeatures,
    NNArchitecture,
)


@dataclass
class ResNetSpec(NNSpec):
    repr_tower_blocks: int = 2
    repr_tower_dim: int = 8
    pred_tower_blocks: int = 2
    pred_tower_dim: int = 8
    dyna_tower_blocks: int = 2
    dyna_tower_dim: int = 8
    dyna_state_blocks: int = 2


@dataclass
class ConvBlock(hk.Module):
    num_channels: int

    def __call__(self, x, is_training):
        x = hk.Conv2D(self.num_channels, (3, 3), padding="same")(x)
        x = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.9)(
            x, is_training=is_training
        )
        x = jax.nn.relu(x)
        return x


@dataclass
class ResBlock(hk.Module):
    def __call__(self, x, is_training):
        orig_x = x
        num_channels = x.shape[-1]
        x = ConvBlock(num_channels)(x, is_training=is_training)
        x = x + orig_x
        x = jax.nn.relu(x)
        return x


@dataclass
class ResTower(hk.Module):
    num_blocks: int

    def __call__(self, x, is_training):
        for _ in range(self.num_blocks):
            x = ResBlock()(x, is_training)
        return x


class ResNetArchitecture(NNArchitecture):
    def __init__(self, spec: ResNetSpec):
        assert isinstance(spec, ResNetSpec), "spec must be of type ResNetSpec"
        super().__init__(spec)

    def _repr_net(self, stacked_frames, is_training):
        height, width, channels = self.spec.stacked_frames_shape
        tower_dim = self.spec.repr_tower_dim
        chex.assert_shape(stacked_frames, (None, height, width, channels))

        hidden_state = ConvBlock(tower_dim)(stacked_frames, is_training)
        chex.assert_shape(hidden_state, (None, height, width, tower_dim))

        hidden_state = ResTower(
            num_blocks=self.spec.repr_tower_blocks,
        )(hidden_state, is_training)
        chex.assert_shape(hidden_state, (None, height, width, tower_dim))

        hidden_state = ConvBlock(self.spec.dim_repr)(hidden_state, is_training)
        chex.assert_shape(hidden_state, (None, height, width, self.spec.dim_repr))

        return hidden_state

    def _pred_net(self, hidden_state, is_training):
        height, width, channels = self.spec.stacked_frames_shape
        chex.assert_shape(hidden_state, (None, height, width, self.spec.dim_repr))

        # pred trunk
        trunk_tower_dim = self.spec.pred_tower_dim
        hidden_state = ConvBlock(trunk_tower_dim)(hidden_state, is_training)
        pred_trunk = ResTower(
            num_blocks=self.spec.pred_tower_blocks,
        )(hidden_state, is_training)
        chex.assert_shape(pred_trunk, (None, height, width, trunk_tower_dim))

        pred_trunk_flat = pred_trunk.reshape((pred_trunk.shape[0], -1))
        chex.assert_shape(pred_trunk_flat, [None, height * width * trunk_tower_dim])

        # pred value head
        value = hk.Linear(output_size=1, name="pred_v")(pred_trunk_flat)
        # TODO: sigmoid only for reward range [-1, 1]
        value = jax.nn.sigmoid(value)
        chex.assert_shape(value, (None, 1))

        # pred policy head
        policy_logits = hk.Linear(output_size=self.spec.dim_action, name="pred_p")(
            pred_trunk_flat
        )
        policy_logits = jax.nn.relu(policy_logits)
        chex.assert_shape(policy_logits, (None, self.spec.dim_action))

        return value, policy_logits

    def _dyna_net(self, hidden_state, action, is_training):
        height, width, channels = self.spec.stacked_frames_shape
        chex.assert_shape(hidden_state, (None, height, width, self.spec.dim_repr))

        # make state-action representation
        # TODO: check correctness action one-hot encoding here
        chex.assert_shape(action, [None])
        action_one_hot = jax.nn.one_hot(action, num_classes=self.spec.dim_action)
        action_one_hot = jnp.expand_dims(
            action_one_hot, axis=[1, 2]
        )  # add height and width dim
        action_one_hot = action_one_hot.tile((1, height, width, 1))
        chex.assert_equal_shape_prefix([hidden_state, action_one_hot], prefix_len=3)

        state_action_repr = jnp.concatenate((hidden_state, action_one_hot), axis=-1)
        chex.assert_shape(
            state_action_repr,
            (None, height, width, self.spec.dim_repr + self.spec.dim_action),
        )

        # dyna trunk
        dyna_trunk = ConvBlock(self.spec.dyna_tower_dim)(state_action_repr, is_training)
        dyna_trunk = ResTower(num_blocks=self.spec.dyna_tower_blocks)(
            dyna_trunk, is_training
        )
        chex.assert_shape(dyna_trunk, (None, height, width, self.spec.dyna_tower_dim))

        # dyna hidden state head
        next_hidden_state = ResTower(
            num_blocks=self.spec.dyna_state_blocks,
        )(dyna_trunk, is_training)
        next_hidden_state = ConvBlock(self.spec.dim_repr)(
            next_hidden_state, is_training
        )
        chex.assert_shape(next_hidden_state, (None, height, width, self.spec.dim_repr))

        # dyna reward head
        dyna_trunk_flat = dyna_trunk.reshape((dyna_trunk.shape[0], -1))
        reward = hk.Linear(output_size=1, name="dyna_reward")(dyna_trunk_flat)
        chex.assert_shape(reward, (None, 1))

        return next_hidden_state, reward

    def root_inference(self, root_feats: RootFeatures, is_training: bool):
        hidden_state = self._repr_net(root_feats.stacked_frames, is_training)
        value, policy_logits = self._pred_net(hidden_state, is_training)
        reward = jnp.zeros_like(value)

        chex.assert_rank([value, reward, policy_logits, hidden_state], [2, 2, 2, 4])

        return NNOutput(
            value=value,
            reward=reward,
            policy_logits=policy_logits,
            hidden_state=hidden_state,
        )

    def trans_inference(self, trans_feats: TransitionFeatures, is_training: bool):
        next_hidden_state, reward = self._dyna_net(
            trans_feats.hidden_state,
            trans_feats.action,
            is_training,
        )
        value, policy_logits = self._pred_net(next_hidden_state, is_training)
        chex.assert_rank(
            [value, reward, policy_logits, next_hidden_state], [2, 2, 2, 4]
        )
        return NNOutput(
            value=value,
            reward=reward,
            policy_logits=policy_logits,
            hidden_state=next_hidden_state,
        )
