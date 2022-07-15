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


def conv_3x3(num_channels):
    return hk.Conv2D(
        num_channels,
        (3, 3),
        padding="same",
        with_bias=False,
    )


def bn():
    return hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.999)


@dataclass
class ConvBlock(hk.Module):
    num_channels: int

    def __call__(self, x, is_training):
        x = conv_3x3(self.num_channels)(x)
        x = bn()(x, is_training=is_training)
        x = jax.nn.relu(x)
        return x


@dataclass
class ResBlock(hk.Module):
    def __call__(self, x, is_training):
        identity = x
        num_channels = x.shape[-1]

        x = conv_3x3(num_channels)(x)
        x = bn()(x, is_training)
        x = jax.nn.relu(x)

        x = conv_3x3(num_channels)(x)
        x = bn()(x, is_training)
        x = x + identity
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
        super().__init__(spec)
        assert isinstance(self.spec, ResNetSpec), "spec must be of type ResNetSpec"
        self.spec: ResNetSpec

    def _repr_net(self, obs: jnp.ndarray, is_training: bool):
        """

        Downsampling described in the paper:

            Specifically, starting with an input observation of resolution 96 x 96 and 128 planes
            (32 history frames of 3 colour channels each, concatenated with the corresponding 32
            actions broadcast to planes), we downsample as follows:

            - 1 convolution with stride 2 and 128 output planes, output resolution 48 x 48
            - 2 residual blocks with 128 planes
            - 1 convolution with stride 2 and 256 output planes, output resolution 24 x 24
            - 3 residual blocks with 256 planes
            - average pooling with stride 2, output resolution 12 x 12
            - 3 residual blocks with 256 planes
            - average pooling with stride 2, output resolution 6 x 6.

            The kernel size is 3 x 3 for all operations.

        """
        tower_dim = self.spec.repr_tower_dim
        chex.assert_shape(
            obs,
            (None, self.spec.obs_rows, self.spec.obs_cols, self.spec.obs_channels),
        )

        x = obs

        # Downsample
        obs_resolution = (self.spec.obs_rows, self.spec.obs_cols)
        repr_resolution = (self.spec.repr_rows, self.spec.repr_cols)
        if obs_resolution != repr_resolution:
            x = hk.Conv2D(128, (3, 3), stride=2, padding="SAME")(x)
            x = ResTower(2)(x, is_training)
            x = hk.Conv2D(256, (3, 3), stride=2, padding="SAME")(x)
            x = ResTower(3)(x, is_training)
            x = hk.AvgPool(window_shape=(3, 3, 1), strides=(2, 2, 1), padding="SAME")(x)
            x = ResTower(3)(x, is_training)
            x = hk.AvgPool(window_shape=(3, 3, 1), strides=(2, 2, 1), padding="SAME")(x)

        chex.assert_shape(x, (None, self.spec.repr_rows, self.spec.repr_cols, None))

        hidden_state = ConvBlock(tower_dim)(x, is_training)
        hidden_state = ResTower(
            num_blocks=self.spec.repr_tower_blocks,
        )(hidden_state, is_training)
        chex.assert_shape(
            hidden_state, (None, self.spec.repr_rows, self.spec.repr_cols, tower_dim)
        )

        hidden_state = ConvBlock(self.spec.repr_channels)(hidden_state, is_training)
        # hidden_state = hk.Conv2D(self.spec.repr_channels, (3, 3), padding="same")(
        #     hidden_state
        # )
        chex.assert_shape(
            hidden_state,
            (None, self.spec.repr_rows, self.spec.repr_cols, self.spec.repr_channels),
        )

        return normalize_hidden_state(hidden_state)

    def _pred_net(self, hidden_state, is_training):
        chex.assert_shape(
            hidden_state,
            (None, self.spec.repr_rows, self.spec.repr_cols, self.spec.repr_channels),
        )

        # pred trunk
        trunk_tower_dim = self.spec.pred_tower_dim
        hidden_state = ConvBlock(trunk_tower_dim)(hidden_state, is_training)
        pred_trunk = ResTower(
            num_blocks=self.spec.pred_tower_blocks,
        )(hidden_state, is_training)
        chex.assert_shape(
            pred_trunk,
            (None, self.spec.repr_rows, self.spec.repr_cols, trunk_tower_dim),
        )

        pred_trunk_flat = pred_trunk.reshape((pred_trunk.shape[0], -1))
        chex.assert_shape(
            pred_trunk_flat,
            [None, self.spec.repr_rows * self.spec.repr_cols * trunk_tower_dim],
        )

        # value head
        value_logits = hk.Linear(
            output_size=self.spec.scalar_transform.dim,
            w_init=hk.initializers.Constant(0),
        )(pred_trunk_flat)
        chex.assert_shape(value_logits, (None, self.spec.scalar_transform.dim))

        # policy head
        policy_logits = hk.Linear(
            output_size=self.spec.dim_action, w_init=hk.initializers.Constant(0)
        )(pred_trunk_flat)
        chex.assert_shape(policy_logits, (None, self.spec.dim_action))

        return value_logits, policy_logits

    def _dyna_net(self, hidden_state, action, is_training):
        chex.assert_shape(
            hidden_state,
            (None, self.spec.repr_rows, self.spec.repr_cols, self.spec.repr_channels),
        )

        # make state-action representation
        # TODO: check correctness action one-hot encoding here
        chex.assert_shape(action, [None])
        action_one_hot = jax.nn.one_hot(action, num_classes=self.spec.dim_action)
        # broadcast to self.spec.repr_rows and self.spec.repr_cols dim
        action_one_hot = jnp.expand_dims(action_one_hot, axis=[1, 2])
        action_one_hot = (
            jnp.tile(action_one_hot, (1, self.spec.repr_rows, self.spec.repr_cols, 1))
            / self.spec.dim_action
        )
        chex.assert_equal_shape_prefix([hidden_state, action_one_hot], prefix_len=3)

        state_action_repr = jnp.concatenate((hidden_state, action_one_hot), axis=-1)
        chex.assert_shape(
            state_action_repr,
            (
                None,
                self.spec.repr_rows,
                self.spec.repr_cols,
                self.spec.repr_channels + self.spec.dim_action,
            ),
        )

        # dyna trunk
        dyna_trunk = ConvBlock(self.spec.dyna_tower_dim)(state_action_repr, is_training)
        dyna_trunk = ResTower(num_blocks=self.spec.dyna_tower_blocks)(
            dyna_trunk, is_training
        )
        dyna_trunk_flat = dyna_trunk.reshape((dyna_trunk.shape[0], -1))
        chex.assert_shape(
            dyna_trunk,
            (None, self.spec.repr_rows, self.spec.repr_cols, self.spec.dyna_tower_dim),
        )

        # hidden state head
        next_hidden_state = ResTower(
            num_blocks=self.spec.dyna_state_blocks,
        )(dyna_trunk, is_training)
        next_hidden_state = ConvBlock(self.spec.repr_channels)(
            next_hidden_state, is_training
        )
        # next_hidden_state = hk.Conv2D(self.spec.repr_channels, (3, 3), padding="same")(
        #     next_hidden_state
        # )
        chex.assert_shape(
            next_hidden_state,
            (None, self.spec.repr_rows, self.spec.repr_cols, self.spec.repr_channels),
        )

        # reward head
        reward_logits = hk.Linear(
            output_size=self.spec.scalar_transform.dim,
            w_init=hk.initializers.Constant(0),
        )(dyna_trunk_flat)

        chex.assert_shape(reward_logits, (None, self.spec.scalar_transform.dim))

        # TODO: check correctness of scaling
        next_hidden_state = normalize_hidden_state(next_hidden_state)
        return next_hidden_state, reward_logits

    def _proj_net(self, hidden_state, is_training):
        chex.assert_shape(
            hidden_state,
            (None, self.spec.repr_rows, self.spec.repr_cols, self.spec.repr_channels),
        )
        hidden_state_flatten = hk.Flatten()(hidden_state)
        x = hidden_state_flatten

        x = hk.Linear(32)(x)
        x = bn()(x, is_training=is_training)
        # x = jax.nn.leaky_relu(x)

        # x = hk.Linear(32)(x)
        # x = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.9)(
        #     x, is_training=is_training
        # )
        x = jax.nn.leaky_relu(x)
        x = hk.Linear(hidden_state_flatten.shape[-1])(x)

        projected = x.reshape(hidden_state.shape)
        return projected


def normalize_hidden_state(hidden_state):
    batch_min = jnp.min(hidden_state, axis=(1, 2, 3), keepdims=True)
    batch_max = jnp.max(hidden_state, axis=(1, 2, 3), keepdims=True)
    hidden_state = (hidden_state - batch_min) / (
        batch_max - batch_min + jnp.array(1e-10)
    )
    return hidden_state
