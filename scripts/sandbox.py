# %%
import numpy as np
import chex
import random
from moozi.core import make_env
from moozi.core.vis import BreakthroughVisualizer, save_gif
from moozi.core.env import GIIEnv, GIIVecEnv, GIIEnvFeed, GIIEnvOut
from moozi.planner import Planner
import jax
import jax.numpy as jnp

# %%
arr = jnp.array(0)
arr = jnp.array(1)
# %%
@jax.jit
def hello(x):
    return jax.lax.select(x + arr > 0, True, False)


# %%
hello

# %%
