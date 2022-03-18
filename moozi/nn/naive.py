import jax
import jax.numpy as jnp
import chex
from dataclasses import dataclass
import haiku as hk

from moozi.nn import (
    NNArchitecture
)


class NaiveArchitecture(NNArchitecture):
    def _repr_net(self, stacked_frames, is_training):
        frames_flat = hk.Flatten()(stacked_frames)
        hidden_state = hk.Linear(
            self.spec.repr_rows * self.spec.repr_cols * self.spec.repr_channels
        )(frames_flat)
        hidden_state = hidden_state.reshape(
            (-1, self.spec.repr_rows, self.spec.repr_cols, self.spec.repr_channels)
        )
        return hidden_state

    def _pred_net(self, hidden_state, is_training):
        hidden_state = hk.Flatten()(hidden_state)
        hidden_state = hk.Linear(
            self.spec.repr_rows * self.spec.repr_cols * self.spec.repr_channels
        )(hidden_state)
        value_head = hk.Linear(output_size=1)(hidden_state)
        policy_head = hk.Linear(output_size=self.spec.dim_action)(hidden_state)
        return value_head, policy_head

    def _dyna_net(self, hidden_state, action, is_training):
        action_one_hot = jax.nn.one_hot(action, num_classes=self.spec.dim_action)

        # broadcast to self.spec.repr_rows and self.spec.repr_cols dim
        action_one_hot = jnp.expand_dims(action_one_hot, axis=[1, 2])
        action_one_hot = action_one_hot.tile(
            (1, self.spec.repr_rows, self.spec.repr_cols, 1)
        )

        state_action_repr = jnp.concatenate((hidden_state, action_one_hot), axis=-1)
        state_action_repr = hk.Flatten()(state_action_repr)
        hidden_state = hk.Linear(
            self.spec.repr_rows * self.spec.repr_cols * self.spec.repr_channels
        )(state_action_repr)
        hidden_state = hidden_state.reshape(
            (-1, self.spec.repr_rows, self.spec.repr_cols, self.spec.repr_channels)
        )
        reward = hk.Linear(output_size=1)(state_action_repr)
        return hidden_state, reward
