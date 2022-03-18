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
        raise NotImplementedError

    def _pred_net(self, hidden_state: jnp.ndarray, is_training: bool):
        raise NotImplementedError

    def _dyna_net(
        self, hidden_state: jnp.ndarray, action: jnp.ndarray, is_training: bool
    ):
        raise NotImplementedError
