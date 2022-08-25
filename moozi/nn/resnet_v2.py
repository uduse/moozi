from functools import partial
from typing import Type
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
from moozi.core.utils import make_frame_planes, make_one_hot_planes


@dataclass
class ResNetV2Spec(NNSpec):
    repr_tower_blocks: int = 6
    pred_tower_blocks: int = 1
    dyna_tower_blocks: int = 1
    dyna_state_blocks: int = 1


def conv_1x1(num_channels):
    return hk.Conv2D(
        num_channels,
        (1, 1),
        padding="same",
        with_bias=False,
    )


def conv_3x3(num_channels):
    return hk.Conv2D(
        num_channels,
        (3, 3),
        padding="same",
        with_bias=False,
    )


def ln() -> hk.LayerNorm:
    return hk.LayerNorm(axis=-1, param_axis=-1, create_scale=True, create_offset=True)


@dataclass
class ResBlockV2(hk.Module):
    def __call__(self, x):
        identity = x
        num_channels = x.shape[-1]

        x = ln()(x)
        x = jax.nn.relu(x)
        x = conv_1x1(num_channels)(x)

        x = ln()(x)
        x = jax.nn.relu(x)
        x = conv_3x3(num_channels)(x)

        x = ln()(x)
        x = jax.nn.relu(x)
        x = conv_1x1(num_channels)(x)

        x = x + identity

        return x


@dataclass
class ResTower(hk.Module):
    num_blocks: int

    def __call__(self, x):
        for _ in range(self.num_blocks):
            x = ResBlockV2()(x)
        return x


class ResNetV2Architecture(NNArchitecture):
    def __init__(self, spec: ResNetV2Spec):
        super().__init__(spec)
        assert isinstance(self.spec, ResNetV2Spec), "spec must be of type ResNetV2Spec"
        self.spec: ResNetV2Spec

    def _repr_net(self, feats: RootFeatures, is_training: bool):
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

        chex.assert_shape(
            feats.frames,
            (
                None,
                self.spec.history_length,
                self.spec.frame_rows,
                self.spec.frame_cols,
                self.spec.frame_channels,
            ),
        )
        chex.assert_shape(feats.actions, (None, self.spec.history_length))
        chex.assert_shape(feats.to_play, (None,))
        chex.assert_type(feats.to_play, jnp.int32)

        x_frames = jax.vmap(make_frame_planes)(feats.frames)
        # downsample
        obs_resolution = (self.spec.frame_rows, self.spec.frame_cols)
        repr_resolution = (self.spec.repr_rows, self.spec.repr_cols)
        while obs_resolution != repr_resolution:
            x_frames = hk.Conv2D(
                self.spec.repr_channels, (2, 2), stride=1, padding="VALID"
            )(x_frames)
            x_frames = ResBlockV2()(x_frames)
            x_frames = hk.AvgPool(
                window_shape=(3, 3, 1), strides=(2, 2, 1), padding="SAME"
            )(x_frames)
        chex.assert_shape(
            x_frames, (None, self.spec.repr_rows, self.spec.repr_cols, None)
        )

        action_planes_maker = jax.vmap(
            partial(
                make_one_hot_planes,
                num_rows=self.spec.repr_rows,
                num_cols=self.spec.repr_cols,
                num_classes=self.spec.dim_action,
            )
        )
        x_actions = action_planes_maker(feats.actions)

        player_planes_maker = jax.vmap(
            partial(
                make_one_hot_planes,
                num_rows=self.spec.repr_rows,
                num_cols=self.spec.repr_cols,
                num_classes=self.spec.num_players,
            )
        )
        x_player = player_planes_maker(feats.to_play.reshape(-1, 1))

        stacked = jnp.concatenate([x_frames, x_actions, x_player], axis=-1)

        hidden_state = conv_1x1(self.spec.repr_channels)(stacked)
        hidden_state = ResTower(self.spec.repr_tower_blocks)(hidden_state)
        chex.assert_shape(
            hidden_state,
            (None, self.spec.repr_rows, self.spec.repr_cols, self.spec.repr_channels),
        )

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
        pred_trunk = ResTower(
            self.spec.pred_tower_blocks,
        )(hidden_state)
        chex.assert_shape(
            pred_trunk,
            (None, self.spec.repr_rows, self.spec.repr_cols, self.spec.repr_channels),
        )

        # TODO: use 1 channel conv?
        pred_trunk_flat = pred_trunk.reshape((pred_trunk.shape[0], -1))
        chex.assert_shape(
            pred_trunk_flat,
            [None, self.spec.repr_rows * self.spec.repr_cols * self.spec.repr_channels],
        )

        # value head
        v_head = hk.Linear(128)(pred_trunk_flat)
        v_head = ln()(v_head)
        v_head = jax.nn.relu(v_head)
        v_head = hk.Linear(
            self.spec.scalar_transform.dim,
            w_init=hk.initializers.Constant(0),
            b_init=hk.initializers.Constant(0),
        )(v_head)
        chex.assert_shape(v_head, (None, self.spec.scalar_transform.dim))

        # policy head
        p_head = hk.Linear(128)(pred_trunk_flat)
        p_head = ln()(p_head)
        p_head = jax.nn.relu(p_head)
        p_head = hk.Linear(
            self.spec.dim_action,
            w_init=hk.initializers.Constant(0),
            b_init=hk.initializers.Constant(0),
        )(p_head)
        chex.assert_shape(p_head, (None, self.spec.dim_action))

        return v_head, p_head

    def _dyna_net(self, feats: TransitionFeatures, is_training: bool):
        hidden_state = feats.hidden_state
        action = feats.action
        chex.assert_shape(
            hidden_state,
            (None, self.spec.repr_rows, self.spec.repr_cols, self.spec.repr_channels),
        )
        chex.assert_shape(action, [None])

        # [B] -> [B, 1] -> [B, H, W, A]
        action_planes_maker = jax.vmap(
            partial(
                make_one_hot_planes,
                num_rows=self.spec.repr_rows,
                num_cols=self.spec.repr_cols,
                num_classes=self.spec.dim_action,
            )
        )
        action_planes = action_planes_maker(action.reshape((-1, 1)))
        action_planes = conv_1x1(self.spec.repr_channels)(action_planes)
        chex.assert_equal_shape([hidden_state, action_planes])

        # dyna trunk
        dyna_trunk = hidden_state + action_planes
        dyna_trunk = ResTower(self.spec.dyna_tower_blocks)(dyna_trunk)
        chex.assert_shape(
            dyna_trunk,
            (None, self.spec.repr_rows, self.spec.repr_cols, self.spec.repr_channels),
        )

        # hidden state head
        next_hidden_state = ResTower(self.spec.dyna_state_blocks)(dyna_trunk)
        chex.assert_shape(
            next_hidden_state,
            (None, self.spec.repr_rows, self.spec.repr_cols, self.spec.repr_channels),
        )

        # reward head
        dyna_trunk_flat = dyna_trunk.reshape((dyna_trunk.shape[0], -1))
        r_head = hk.Linear(128)(dyna_trunk_flat)
        r_head = ln()(r_head)
        r_head = jax.nn.relu(r_head)
        r_head = hk.Linear(
            self.spec.scalar_transform.dim,
            w_init=hk.initializers.Constant(0),
            b_init=hk.initializers.Constant(0),
        )(r_head)
        chex.assert_shape(r_head, (None, self.spec.scalar_transform.dim))

        next_hidden_state = normalize_hidden_state(next_hidden_state)
        return next_hidden_state, r_head

    def _proj_net(self, hidden_state, is_training):
        chex.assert_shape(
            hidden_state,
            (None, self.spec.repr_rows, self.spec.repr_cols, self.spec.repr_channels),
        )
        projected = ResTower(1)(hidden_state)
        return projected


def normalize_hidden_state(hidden_state, kind='unit'):
    if kind == 'unit':
        batch_min = jnp.min(hidden_state, axis=(1, 2, 3), keepdims=True)
        batch_max = jnp.max(hidden_state, axis=(1, 2, 3), keepdims=True)
        hidden_state = (hidden_state - batch_min) / (
            batch_max - batch_min + jnp.array(1e-12)
        )
        return hidden_state
    elif kind == 'normal':
        batch_mean = jnp.mean(hidden_state, axis=(1, 2, 3), keepdims=True)
        batch_std = jnp.std(hidden_state, axis=(1, 2, 3), keepdims=True)
        hidden_state = (hidden_state - batch_mean) / (batch_std + 1e-4)
        return hidden_state