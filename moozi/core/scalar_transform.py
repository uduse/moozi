from typing import Callable, NamedTuple
from flax import struct

import chex
import jax
import jax.numpy as jnp
import numpy as np


class ScalarTransform(struct.PyTreeNode):
    support_min: int
    support_max: int
    dim: int
    epsilon: float
    scalar_min: float
    scalar_max: float
    transform: Callable[[chex.Array], chex.Array]
    inverse_transform: Callable[[chex.Array], chex.Array]
    contract: bool

    @staticmethod
    def new(
        support_min: int,
        support_max: int,
        eps: float = 1e-3,
        contract: bool = True,
    ) -> "ScalarTransform":
        return make_scalar_transform(support_min, support_max, eps, contract)


def _phi(scalar, eps):
    return jnp.sign(scalar) * (jnp.sqrt(jnp.abs(scalar) + 1) - 1) + eps * scalar


def _inverse_phi(scalar, eps):
    return jnp.sign(scalar) * (
        jnp.power(
            ((jnp.sqrt(1 + 4 * eps * (jnp.abs(scalar) + 1 + eps)) - 1) / (2 * eps)),
            2,
        )
        - 1
    )


# TODO: merge this into the smart constructor
def make_scalar_transform(
    support_min: int,
    support_max: int,
    eps: float = 1e-3,
    contract: bool = True,
) -> ScalarTransform:
    support_dim = support_max - support_min + 1

    def _scalar_transform(scalar: chex.Scalar) -> chex.Scalar:
        if contract:
            scalar = _phi(scalar, eps)
        scalar = jnp.clip(scalar, support_min, support_max)
        lower_val = jnp.floor(scalar).astype(jnp.int32)
        upper_val = jnp.ceil(scalar + np.finfo(np.float32).eps).astype(jnp.int32)
        lower_factor = upper_val - scalar
        upper_factor = 1 - lower_factor
        lower_idx = lower_val - support_min
        upper_idx = upper_val - support_min
        probs = jnp.zeros(support_dim)
        probs = probs.at[lower_idx].set(lower_factor)
        probs = probs.at[upper_idx].set(upper_factor)
        chex.assert_shape(probs, (support_dim,))
        return probs

    def _inverse_scalar_transform(probs: chex.Array) -> chex.Array:
        chex.assert_shape(probs, (support_dim,))
        support_vals = jnp.arange(support_min, support_max + 1, dtype=jnp.float32)
        scalar = jnp.sum(probs * support_vals)
        if contract:
            scalar = _inverse_phi(scalar, eps)
        return scalar

    lower = jax.nn.one_hot(0, num_classes=support_dim, dtype=float)
    scalar_min = _inverse_scalar_transform(lower)
    upper = jax.nn.one_hot(support_dim - 1, num_classes=support_dim, dtype=float)
    scalar_max = _inverse_scalar_transform(upper)

    return ScalarTransform(
        support_min,
        support_max,
        support_dim,
        eps,
        scalar_min,
        scalar_max,
        jax.vmap(_scalar_transform),
        jax.vmap(_inverse_scalar_transform),
        contract=contract,
    )
