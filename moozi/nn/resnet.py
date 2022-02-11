import jax
import jax.numpy as jnp
import chex
from dataclasses import dataclass
import haiku as hk

from moozi.nn import (
    NNOutput,
    NNSpec,
    RootInferenceFeatures,
    TransitionInferenceFeatures,
)


class ConvBlock(hk.Module):
    def __call__(self, x, is_training):
        x = hk.Conv2D(16, (3, 3), padding="same")(x)
        x = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.9)(
            x, is_training=is_training
        )
        x = jax.nn.relu(x)
        return x


@dataclass
class ResBlock(hk.Module):
    output_channels: int = 16

    def __call__(self, x, is_training):
        orig_x = x
        x = hk.Conv2D(
            output_channels=self.output_channels, kernel_shape=(3, 3), padding="same"
        )(x)
        x = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.9)(
            x, is_training=is_training
        )
        x = jax.nn.relu(x)
        x = hk.Conv2D(
            output_channels=self.output_channels, kernel_shape=(3, 3), padding="same"
        )(x)
        x = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.9)(
            x, is_training=is_training
        )
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


# TODO: merge two res towers
@dataclass
class ResTowerV2(hk.Module):
    num_blocks: int
    res_channels: int
    output_channels: int

    def __call__(self, x, is_training):
        for _ in range(self.num_blocks):
            x = ResBlock(output_channels=self.res_channels)(x, is_training)
        x = hk.Conv2D(
            output_channels=self.output_channels, kernel_shape=(3, 3), padding="same"
        )(x)
        return x


class ResNetArchitecture(hk.Module):
    def __init__(self, spec: NNSpec):
        super().__init__()
        self.spec = spec

    def _repr_net(self, stacked_frames, is_training):
        chex.assert_rank(
            stacked_frames, 5
        )  # (batch_size, num_frames, height, width, channels)
        # TODO: make stacked frames store like this by default so we don't have to do the transpose here
        stacked_frames = stacked_frames.transpose(0, 2, 3, 4, 1)
        stacked_frames = stacked_frames.reshape(
            stacked_frames.shape[:-2] + (-1,)
        )  # stack num_frames and channels as features planes

        hidden_state = ConvBlock()(stacked_frames, is_training)
        hidden_state = ResTower(num_blocks=self.spec.extra["repr_net_num_blocks"])(
            hidden_state, is_training
        )
        hidden_state = hk.Conv2D(
            output_channels=self.spec.dim_repr, kernel_shape=(3, 3), padding="same"
        )(hidden_state)

        chex.assert_rank(hidden_state, 4)  # (batch_size, height, width, dim_repr)

        return hidden_state

    def _pred_net(self, hidden_state, is_training):
        pred_trunk = ResTower(num_blocks=self.spec.extra["pred_trunk_num_blocks"])(
            hidden_state, is_training
        )
        pred_trunk = hk.Conv2D(
            output_channels=self.spec.dim_repr, kernel_shape=(3, 3), padding="same"
        )(pred_trunk)

        chex.assert_rank(pred_trunk, 4)  # (batch_size, height, width, dim_repr)

        pred_trunk_flat = pred_trunk.reshape((pred_trunk.shape[0], -1))
        chex.assert_rank(pred_trunk_flat, 2)  # (batch_size, height * width * dim_repr)

        value = hk.Linear(output_size=1, name="pred_v")(pred_trunk_flat)
        chex.assert_shape(value, (None, 1))  # (batch_size, 1)

        policy_logits = hk.Linear(output_size=self.spec.dim_action, name="pred_p")(
            pred_trunk_flat
        )
        chex.assert_shape(
            policy_logits, (None, self.spec.dim_action)
        )  # (batch_size, dim_action)

        return value, policy_logits

    def _dyna_net(self, hidden_state, action, is_training):
        action_one_hot = jax.nn.one_hot(action, num_classes=self.spec.dim_action)
        action_one_hot = action_one_hot.tile(hidden_state.shape[0:3] + (1,))

        chex.assert_equal_rank([hidden_state, action_one_hot])
        state_action_repr = jnp.concatenate((hidden_state, action_one_hot), axis=-1)

        dyna_trunk = ResTowerV2(
            num_blocks=self.spec.extra["dyna_trunk_num_blocks"],
            res_channels=state_action_repr.shape[-1],
            output_channels=self.spec.dim_repr,
        )(state_action_repr, is_training)

        next_hidden_state = ResTowerV2(
            num_blocks=self.spec.extra["dyna_hidden_num_blocks"],
            res_channels=self.spec.dim_repr,
            output_channels=self.spec.dim_repr,
        )(dyna_trunk, is_training)

        reward = hk.Linear(output_size=1, name="dyna_reward")(
            dyna_trunk.reshape((dyna_trunk.shape[0], -1))
        )
        chex.assert_shape(reward, (None, 1))  # (batch_size, 1)

        return next_hidden_state, reward

    def initial_inference(
        self, init_inf_feats: RootInferenceFeatures, is_training: bool
    ):
        hidden_state = self._repr_net(init_inf_feats.stacked_frames, is_training)
        value, policy_logits = self._pred_net(hidden_state, is_training)
        reward = jnp.zeros_like(value)

        chex.assert_rank([value, reward, policy_logits, hidden_state], [2, 2, 2, 4])

        return NNOutput(
            value=value,
            reward=reward,
            policy_logits=policy_logits,
            hidden_state=hidden_state,
        )

    def recurrent_inference(
        self, recurr_inf_feats: TransitionInferenceFeatures, is_training: bool
    ):
        next_hidden_state, reward = self._dyna_net(
            recurr_inf_feats.hidden_state,
            recurr_inf_feats.action,
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
