# %%
import ray
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

from lib import get_config, get_model, training_suite_factory

# %%
config = get_config()
model = get_model(config)
num_envs = 4
stacker = HistoryStacker(
    num_rows=config.env.num_rows,
    num_cols=config.env.num_cols,
    num_channels=config.env.num_channels,
    history_length=config.history_length,
    dim_action=config.dim_action,
)
vis = BreakthroughVisualizer(num_rows=config.env.num_rows, num_cols=config.env.num_cols)

# %%
rb = ray.remote(ReplayBuffer).remote(**config.replay.kwargs)
ps = ray.remote(num_gpus=config.param_opt.num_gpus)(ParameterServer).remote(
    training_suite_factory(config), use_remote=True
)
tw = TrainingWorker(
    index=0,
    env_name=config.env.name,
    num_envs=num_envs,
    model=model,
    stacker=stacker,
    planner=Planner(
        batch_size=num_envs,
        dim_action=config.dim_action,
        model=model,
        discount=config.discount,
        num_unroll_steps=config.num_unroll_steps,
        num_simulations=1,
        limit_depth=True,
    ),
    num_steps=50,
)

# %%
def train():
    for i in range(1000):
        trajs = tw.run()
        rb.add_trajs(trajs)
        batch = rb.get_train_targets_batch(1024)
        loss = ps.update(batch, 256)
        ps.log_tensorboard(i)
        if i % 10 == 0:
            ps.save()
        print(f"{loss=}")


# %%
def load_params_and_states(checkpoints_path="checkpoints") -> dict:
    ret = {}
    for fpath in Path(checkpoints_path).iterdir():
        name = int(fpath.stem)
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
    model=model,
    discount=-1.0,
    num_unroll_steps=3,
    num_simulations=10,
    limit_depth=False,
)
gii = GII(
    config.env.name,
    stacker=stacker,
    planner=planner,
    params=None,
    state=None,
    random_key=jax.random.PRNGKey(0),
)

# %%
# candidates = [
#     Candidate(name=name, params=params, state=state, planner=planner, elo=1300)
#     for name, (params, state) in load_params_and_states().items()
# ]
# candidates = [c for i, c in enumerate(candidates) if i % 10 == 0]
# print(len(candidates))
# t = Tournament(
#     gii=gii,
#     num_matches=1,
#     candidates=candidates,
# )
# t.run()

# %%
lookup = load_params_and_states("/home/zeyi/moozi/examples/open_spiel/checkpoints")
latest = list(lookup.keys())[-1]
gii.params = {0: lookup[latest][0], 1: lookup[latest][0]}
gii.state = {0: lookup[latest][1], 1: lookup[latest][1]}

# %%
for i in range(9):
    gii.tick()
# jax.config.update("jax_disable_jit", True)
gii.tick()

# %%
search_tree = gii.planner_out.tree
root_state = gii.env.envs[0].backend.get_state
vis_dir = Path("~/assets/search").expanduser()
g = visualize_search_tree(vis, search_tree, root_state, vis_dir)

# # %%
from IPython import display

display.Image(vis_dir / "search.png")

# # %%

# %%
# %%
# %%
# %%
# %%
# %%


# %%
# rb.get_stats()
# # %%
# aei.tick()
# display(tw.vis.make_image(aei.env_out.frame[0]))
# backend_states.append(aei.env.envs[0]._backend.get_state)
# search_trees.append(aei.planner_out.tree)

# # %%
# # %%
# idx = -1
# root_state = backend_states[idx]
# search_tree = search_trees[idx]
# image_path = Path("/home/zeyi/assets/imgs")
# for key, game_state in node_states.items():
#     save_state_to_image(tw.vis, game_state, key)

# # %%
# g = convert_tree_to_graph(search_tree, image_path=str(image_path))
# g.write("/home/zeyi/assets/graph.dot")

# # %%
import ray

@ray.remote
def put():
    return ray.put(1)
    
@ray.remote
def get(x):
    print(x)

x = put.remote()
get.remote(x)
