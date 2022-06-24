# %%
import random
import tree
import jax
import moozi as mz
from moozi.core.link import link, unlink
from moozi.core.tape import include
from moozi.laws import (
    Law,
    get_keys,
    make_vec_env,
    make_min_atar_gif_recorder,
    sequential,
)
import jax.numpy as jnp
import numpy as np

# %%
def _termination_penalty(is_last, reward):
    reward_overwrite = jax.lax.cond(
        is_last,
        lambda: reward - 1,
        lambda: reward,
    )
    return {"reward": reward_overwrite}


penalty = Law.from_fn(_termination_penalty)
f = jax.vmap(unlink(penalty.apply))

# %%
f(jnp.array([True, False]))


# %%
