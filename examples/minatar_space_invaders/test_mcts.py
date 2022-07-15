# %%
from acme.jax.utils import add_batch_dim
from acme.utils.tree_utils import stack_sequence_fields, unstack_sequence_fields
from omegaconf import OmegaConf
import numpy as np
from IPython.display import display
from loguru import logger
import ray
from functools import partial
from moozi.laws import MinAtarVisualizer
from moozi.nn.training import make_target_from_traj, _make_obs_from_train_target

from moozi.replay import ReplayBuffer
from moozi.parameter_optimizer import ParameterServer
from moozi.rollout_worker import RolloutWorker
from moozi.laws import *
from moozi.planner import convert_tree_to_graph, make_gumbel_planner
from moozi.nn import RootFeatures, TransitionFeatures

from lib import (
    training_suite_factory,
    make_test_worker_universe,
    make_reanalyze_universe,
    make_env_worker_universe,
    config,
    model,
    scalar_transform,
)

# %%
vec_env = make_vec_env("MinAtar:SpaceInvaders-v1", num_envs=1)

# %%
config = OmegaConf.load(Path(__file__).parent / "config.yml")

# config.debug = True
config.train.env_worker.num_workers = 1
config.train.env_worker.num_envs = 1
OmegaConf.resolve(config)
print(OmegaConf.to_yaml(config, resolve=True))
OmegaConf.resolve(config)

# %%
ps = ParameterServer(training_suite_factory=training_suite_factory(config))
rb = ReplayBuffer(**config.replay)
vis = MinAtarVisualizer()

# %%
weights_path = "/home/zeyi/miniconda3/envs/moozi/.guild/runs/561ace079fe84e9a9dc39944138f05f3/checkpoints/5900.pkl"
ps.restore(weights_path)

# %%
u = make_env_worker_universe(config)
u.tape["params"] = ps.get_params()
u.tape["state"] = ps.get_state()

# %%
for i in range(30):
    print(i)
    u.tick()

    image = vis.make_image(u.tape["frame"][0])
    image = vis.add_descriptions(
        image,
        action=u.tape["action"][0],
        q_values=u.tape["q_values"][0],
        action_probs=u.tape["action_probs"][0],
        prior_probs=u.tape["prior_probs"][0],
        root_value=u.tape["root_value"][0],
        reward=u.tape["reward"][0],
        visit_counts=u.tape["visit_counts"][0],
    )
    display(image)
    graph = convert_tree_to_graph(u.tape["tree"])
    graph.draw(f"/tmp/graph_{i}.dot", prog="dot")
