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

rng_key = jax.random.PRNGKey(0)

# %% 
num_actions = 1000
legal_actions = jnp.ones(num_actions)
# legal_actions = jnp.zeros(num_actions)
# legal_actions = legal_actions.at[:10].set(1)
dirichlet_alpha = 0.3
batch_size = 1
noise = jax.random.dirichlet(
    rng_key,
    alpha=jnp.full([num_actions], fill_value=dirichlet_alpha) * legal_actions.at[10:].set(0),
    shape=(batch_size,))
rng_key, _ = jax.random.split(rng_key)
np.round(noise * 0.25, 3) * legal_actions.at[10:].set(0)

# %%
