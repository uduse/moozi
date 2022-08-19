# %%
from moozi.core.vis import BreakthroughVisualizer, save_gif
from moozi.core.env import GIIEnv, GIIVecEnv, GIIEnvFeed, GIIEnvOut
from moozi.nn import RootFeatures, NNModel
from moozi.nn.training import make_target_from_traj
from moozi.planner import Planner
from moozi.replay import ReplayBuffer
from moozi.parameter_optimizer import ParameterServer
from moozi.core import HistoryStacker, TrajectoryCollector
from moozi.training_worker import TrainingWorker
from lib import get_model, get_config, training_suite_factory

from typing import Callable, Dict, Tuple, Optional, List, Union, Any
import haiku as hk

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

# %%
rb = ReplayBuffer(**config.replay.kwargs)
ps = ParameterServer(training_suite_factory(config))
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
for i in range(50):
    trajs = tw.run()
    rb.add_trajs(trajs)
    batch = rb.get_train_targets_batch(1024)
    loss = ps.update(batch, 256)
    ps.log_tensorboard(i)
    if i % 10 == 0:
        ps.save()
    print(f"{loss=}")


# %% 
# %% 
# %% 
# %% 
# %% 
# %% 
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
