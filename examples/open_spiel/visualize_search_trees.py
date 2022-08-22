# %%
from loguru import logger
import numpy as np
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cloudpickle
import haiku as hk
import jax
from moozi.core import HistoryStacker, TrajectoryCollector
from moozi.core.env import GIIEnv, GIIEnvFeed, GIIEnvOut, GIIVecEnv
from moozi.core.types import TrainingState
from moozi.core.vis import BreakthroughVisualizer, visualize_search_tree
from moozi.gii import GII
from moozi.nn import NNModel, RootFeatures
from moozi.nn.training import make_target_from_traj
from moozi.parameter_optimizer import ParameterServer, load_params_and_state
from moozi.planner import Planner
from moozi.replay import ReplayBuffer
from moozi.tournament import Player, Tournament
from moozi.training_worker import TrainingWorker
from moozi.driver import Driver, get_config

# %%
logger.info("Loading config")
config = get_config()
driver = Driver.setup(config)
vis = BreakthroughVisualizer(num_rows=config.env.num_rows, num_cols=config.env.num_cols)

# %%
planner = Planner(
    batch_size=1,
    dim_action=config.vis.planner.dim_action,
    model=driver.model,
    discount=config.vis.planner.discount,
    num_simulations=config.vis.planner.num_simulations,
    max_depth=config.vis.planner.max_depth,
    kwargs=config.vis.planner.kwargs
)
logger.info(f"Using planner {planner}")
logger.info("Loading checkpoints")
params, state = load_params_and_state(config.vis.checkpoint_path)
gii = GII(
    config.env.name,
    stacker=driver.stacker,
    planner=planner,
    params=params,
    state=state,
    random_key=jax.random.PRNGKey(0),
)

for i in range(config.vis.num_steps):
    gii.tick()
    search_tree = gii.planner_out.tree
    search_tree = jax.tree_util.tree_map(lambda x: np.asarray(x), search_tree)
    root_state = gii.env.envs[0].backend.get_state
    vis_dir = Path(f"search_tree/{i}").expanduser()
    g = visualize_search_tree(vis, search_tree, root_state, vis_dir)

del g
