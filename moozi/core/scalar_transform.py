from typing import Callable, NamedTuple

import chex
import jax
import jax.numpy as jnp
import numpy as np


class ScalarTransform(NamedTuple):
    support_min: int
    support_max: int
    epsilon: float
    transform: Callable[[chex.Array], chex.Array]
    inverse_transform: Callable[[chex.Array], chex.Array]


def make_scalar_transform(support_min, support_max, eps=1e-3):
    def scalar_transform(scalar: float):
        scalar = jnp.sign(scalar) * (jnp.sqrt(jnp.abs(scalar) + 1) - 1) + eps * scalar
        support_dim = support_max - support_min + 1
        scalar = jnp.clip(scalar, support_min, support_max)
        lower_val = jnp.floor(scalar).astype(jnp.int32)
        upper_val = jnp.ceil(scalar + np.finfo(np.float32).eps).astype(jnp.int32)
        lower_factor = upper_val - scalar
        upper_factor = 1 - lower_factor
        lower_idx = lower_val - support_min
        upper_idx = upper_val - support_min
        vec = jnp.zeros(support_dim)
        vec = vec.at[lower_idx].set(lower_factor)
        vec = vec.at[upper_idx].set(upper_factor)
        chex.assert_shape(vec, (support_dim,))
        return vec

    def inverse_scalar_transform(
        vec: np.ndarray,
    ):
        support_dim = support_max - support_min + 1
        chex.assert_shape(vec, (support_dim,))
        support_vals = jnp.arange(support_min, support_max + 1, dtype=jnp.float32)
        val = jnp.sum(vec * support_vals)
        return jnp.sign(val) * (
            jnp.power(
                ((jnp.sqrt(1 + 4 * eps * (jnp.abs(val) + 1 + eps)) - 1) / (2 * eps)),
                2,
            )
            - 1
        )

    scalar_transform_batched = jax.vmap(scalar_transform)
    inverse_scalar_transform_batched = jax.vmap(inverse_scalar_transform)

    return ScalarTransform(
        support_min,
        support_max,
        eps,
        scalar_transform_batched,
        inverse_scalar_transform_batched,
    )
