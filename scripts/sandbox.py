# %%
from dotenv import load_dotenv
from moozi.driver import Driver, get_config
import numpy as np
import chex
import random
from moozi.core import _make_dm_env
from moozi.core.vis import BreakthroughVisualizer, save_gif
from moozi.core.env import GIIEnv, GIIVecEnv, GIIEnvFeed, GIIEnvOut
from moozi.planner import Planner
import jax
import jax.numpy as jnp
from moozi.replay import ReplayBuffer
from moozi.driver import ConfigFactory
from moozi.gii import GII
from moozi.training_worker import TrainingWorker
import pyspiel

rng_key = jax.random.PRNGKey(0)

load_dotenv()
config = get_config("/home/zeyi/moozi/examples/minatar/config.yml")
factory = ConfigFactory(config)

# %%


# %%

# %%

# %%

# %%

#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# GRAVEYARD
# GRAVEYARD
# GRAVEYARD
# GRAVEYARD
# GRAVEYARD
# GRAVEYARD
# GRAVEYARD
# GRAVEYARD
# GRAVEYARD
# GRAVEYARD
# %%

# # %%
num_actions = 100
legal_actions = jnp.zeros(num_actions).at[:13].set(1)
# legal_actions = jnp.zeros(num_actions)
# legal_actions = legal_actions.at[:10].set(1)
dirichlet_alpha = 0.5
batch_size = 1
alpha = jnp.where(
    jnp.logical_not(legal_actions),
    0,
    jnp.full([num_actions], fill_value=dirichlet_alpha),
)
# alpha = jnp.full([num_actions], fill_value=dirichlet_alpha)
noise = jax.random.dirichlet(rng_key, alpha=alpha, shape=(batch_size,))
rng_key, _ = jax.random.split(rng_key)
np.round(jnp.where(legal_actions, noise * 0.25, 0), 3)

# %%
graph = pygraphviz.AGraph(directed=True, imagepath=image_dir, dpi=72)
graph.node_attr.update(
    imagescale=True,
    shape="box",
    imagepos="tc",
    fixed_size=True,
    labelloc="b",
    fontname="Courier New",
)
graph.edge_attr.update(fontname="Courier New")

graph.add_node(0, label="some label", width=2.5, height=4, image="image_path")
# Add all other nodes and connect them up.
for i, node in enumerate(nodes):
    graph.add_node(
        child_id, label="some other label", width=2.5, height=4, image="image_path"
    )
