# %%
import random
import tree
import jax
import moozi as mz
from moozi.core.link import link, unlink
from moozi.core.scalar_transform import make_scalar_transform
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
st = make_scalar_transform(-10, 10)
vals = np.arange(10)
print(vals.shape)
st.transform(vals).shape
# %%
