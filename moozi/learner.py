import typing
import haiku as hk

import acme
import acme.jax.utils
import chex
import jax
import jax.numpy as jnp
import optax
import rlax

import moozi as mz


class TrainingState(typing.NamedTuple):
    params: hk.Params
    target_params: hk.Params
    state: hk.State
    opt_state: optax.OptState
    steps: int
    rng_key: jax.random.KeyArray
