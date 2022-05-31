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
class MLPSpec(NNSpec):
    pass


class MLPArchitecture(NNArchitecture):
    def __init__(self, spec: MLPSpec):
        assert isinstance(spec, MLPSpec), "spec must be of type ResNetSpec"
        super().__init__(spec)

    def _repr_net(self, stacked_frames: jnp.ndarray, is_training: bool):
        frames_flat = hk.Flatten()(stacked_frames)
        # define mlp structure
        mlp = hk.Sequential(
            [
                hk.Linear(32),
                jax.nn.relu,
                hk.Linear(32),
                jax.nn.relu,
                hk.Linear(
                    self.spec.repr_rows * self.spec.repr_cols * self.spec.repr_channels
                ),
            ]
        )

        # apply mlp
        hidden_state = mlp(frames_flat)
        hidden_state = hidden_state.reshape(
            (-1, self.spec.repr_rows, self.spec.repr_cols, self.spec.repr_channels)
        )
        return hidden_state

    def _pred_net(self, hidden_state: jnp.ndarray, is_training: bool):
        hidden_state = hk.Flatten()(hidden_state)
        # common structure for feature extraction
        mlp = hk.Sequential(
            [
                hk.Linear(32),
                jax.nn.relu,
                hk.Linear(32),
                jax.nn.relu,
                hk.Linear(
                    self.spec.repr_rows * self.spec.repr_cols * self.spec.repr_channels
                ),
            ]
        )
        hidden_state = mlp(hidden_state)
        # value predition
        value_head = hk.Linear(output_size=1)(hidden_state)
        # prior policy prediction
        policy_head = hk.Linear(output_size=self.spec.dim_action)(hidden_state)

        return value_head, policy_head

    def _dyna_net(
        self, hidden_state: jnp.ndarray, action: jnp.ndarray, is_training: bool
    ):
        # create action tiling
        action_one_hot = jax.nn.one_hot(action, num_classes=self.spec.dim_action)
        action_one_hot = jnp.expand_dims(action_one_hot, axis=[1, 2])
        action_one_hot = jnp.tile(
            action_one_hot, (1, self.spec.repr_rows, self.spec.repr_cols, 1)
        )
        # state-action representation
        state_action_repr = jnp.concatenate((hidden_state, action_one_hot), axis=-1)
        state_action_repr = hk.Flatten()(state_action_repr)

        # common component for feature extraction
        mlp = hk.Sequential([hk.Linear(32), jax.nn.relu, hk.Linear(32), jax.nn.relu])

        feature = mlp(state_action_repr)

        # generate next hidden_state
        hidden_state = hk.Linear(
            self.spec.repr_rows * self.spec.repr_cols * self.spec.repr_channels
        )(feature)
        hidden_state = hidden_state.reshape(
            (-1, self.spec.repr_rows, self.spec.repr_cols, self.spec.repr_channels)
        )

        # predict reward
        reward = hk.Linear(output_size=1)(feature)

        return hidden_state, reward
