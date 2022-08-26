import jax
import jax.numpy as jnp
from dataclasses import dataclass
import haiku as hk

from moozi.core.utils import make_one_hot_planes
from moozi.nn import NNArchitecture, RootFeatures, TransitionFeatures


class NaiveArchitecture(NNArchitecture):
    def _repr_net(self, feats: RootFeatures, is_training: bool):
        frames_flat = hk.Flatten()(feats.frames)
        player_flat = hk.Flatten()(jax.nn.one_hot(feats.to_play, num_classes=2))
        cat = jnp.concatenate((frames_flat, player_flat), axis=-1)
        hidden_state = hk.Linear(
            self.spec.repr_rows * self.spec.repr_cols * self.spec.repr_channels
        )(cat)
        hidden_state = hidden_state.reshape(
            (-1, self.spec.repr_rows, self.spec.repr_cols, self.spec.repr_channels)
        )
        return hidden_state

    def _pred_net(self, hidden_state, is_training):
        hidden_state = hk.Flatten()(hidden_state)
        hidden_state = hk.Linear(
            self.spec.repr_rows * self.spec.repr_cols * self.spec.repr_channels
        )(hidden_state)
        value_head = hk.Linear(output_size=self.spec.scalar_transform.dim)(hidden_state)
        policy_head = hk.Linear(output_size=self.spec.dim_action)(hidden_state)
        return value_head, policy_head

    def _dyna_net(self, feats: TransitionFeatures, is_training: bool):
        action_one_hot = jax.nn.one_hot(feats.action, num_classes=self.spec.dim_action)
        hidden_state = hk.Flatten()(feats.hidden_state)
        state_action_repr = jnp.concatenate((hidden_state, action_one_hot), axis=-1)
        state_action_repr = hk.Flatten()(state_action_repr)
        hidden_state = hk.Linear(
            self.spec.repr_rows * self.spec.repr_cols * self.spec.repr_channels
        )(state_action_repr)
        hidden_state = hidden_state.reshape(
            (-1, self.spec.repr_rows, self.spec.repr_cols, self.spec.repr_channels)
        )
        reward = hk.Linear(output_size=self.spec.scalar_transform.dim)(state_action_repr)
        return hidden_state, reward

    def _proj_net(self, hidden_state, is_training):
        return hk.Conv2D(output_channels=hidden_state.shape[-1], kernel_shape=(1, 1))(hidden_state)
