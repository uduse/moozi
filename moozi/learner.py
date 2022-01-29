import typing

import acme
import acme.jax.utils
import chex
import jax
import jax.numpy as jnp
import optax
import rlax

import moozi as mz


class TrainingState(typing.NamedTuple):
    params: chex.ArrayTree
    target_params: chex.ArrayTree
    opt_state: optax.OptState
    steps: int
    rng_key: jax.random.KeyArray
