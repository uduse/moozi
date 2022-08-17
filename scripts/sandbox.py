# %%
import numpy as np
import chex
import random
from moozi.core import make_env
from moozi.core.vis import BreakthroughVisualizer, save_gif
from moozi.core.env import GIIEnv, GIIVecEnv, GIIEnvFeed, GIIEnvOut
from moozi.planner import Planner

# %%
env = GIIEnv("OpenSpiel:breakthrough(rows=5,columns=6)")
num_rows, num_cols = env._backend.observation_spec().observation.shape[:2]
vis = BreakthroughVisualizer(num_rows, num_cols)

feed = env.init()
imgs = []
for i in range(100):
    env_out = env.step(feed)
    feed.action = np.random.choice(np.argwhere(env_out.legal_actions).ravel())
    feed.reset = env_out.is_last
    img = vis.make_image(env_out.frame)
    imgs.append(img)
save_gif(imgs)

# %%
env = GIIVecEnv("OpenSpiel:breakthrough(rows=5,columns=6)", num_envs=16)

# %%
@chex.dataclass
class AgentFeed:
    env_timestep: GIIEnvOut

# %% 
@chex.dataclass
class Agent:
    planner: Planner
    
    def step(self):
        self.planner()

# %%
agent = Agent(
    planner=Planner(
        batch_size,
        dim_action,
        model,
        discount,
        num_unroll_steps,
        num_simulations,
        limit_depth,
        output_tree,

    )
)
