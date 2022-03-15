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


class NaiveArchitecture(NNArchitecture):
    def _repr_net(self, stacked_frames, is_training):
        pass

    def _pred_net(self, hidden_state, is_training):
        pass

    def _dyna_net(self, hidden_state, action, is_training):
        pass
