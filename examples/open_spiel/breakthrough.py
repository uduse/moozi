# %%
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
from moozi.parameter_optimizer import ParameterServer
from moozi.planner import Planner
from moozi.replay import ReplayBuffer
from moozi.tournament import Candidate, Tournament
from moozi.training_worker import TrainingWorker

from lib import get_config
from train import Driver

# %%
config = get_config()
driver = Driver.setup(config)
vis = BreakthroughVisualizer(num_rows=config.env.num_rows, num_cols=config.env.num_cols)

# %%
def load_params_and_states(checkpoints_path="checkpoints") -> dict:
    ret = {}
    for fpath in Path(checkpoints_path).iterdir():
        name = fpath.stem
        training_state: TrainingState
        with open(fpath, "rb") as f:
            _, training_state, _ = cloudpickle.load(f)
        params, state = training_state.target_params, training_state.target_state
        ret[name] = (params, state)
    return ret


# %%
planner = Planner(
    batch_size=1,
    dim_action=config.dim_action,
    model=driver.model,
    discount=-1.0,
    num_unroll_steps=3,
    num_simulations=50,
    limit_depth=True,
)
gii = GII(
    config.env.name,
    stacker=driver.stacker,
    planner=planner,
    params=None,
    state=None,
    random_key=jax.random.PRNGKey(0),
)

# %%
lookup = load_params_and_states("/home/zeyi/moozi/examples/open_spiel/checkpoints")
latest = lookup['latest']
gii.params = {0: latest[0], 1: latest[0]}
gii.state = {0: latest[1], 1: latest[1]}
# %%
for i in range(20):
    gii.tick()
    search_tree = gii.planner_out.tree
    search_tree = jax.tree_util.tree_map(lambda x: np.asarray(x), search_tree)
    root_state = gii.env.envs[0].backend.get_state
    vis_dir = Path(f"~/assets/search_tree/{i}").expanduser()
    g = visualize_search_tree(vis, search_tree, root_state, vis_dir)

# %%
from IPython import display
display.Image(vis_dir / "search.png")