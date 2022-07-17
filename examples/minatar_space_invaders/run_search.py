# %%
from acme.jax.utils import add_batch_dim
from acme.utils.tree_utils import stack_sequence_fields, unstack_sequence_fields
from omegaconf import OmegaConf
import numpy as np
from IPython.display import display
from loguru import logger
import ray
from functools import partial
from moozi.core.tape import make_tape
from moozi.laws import MinAtarVisualizer
from moozi.nn.training import make_target_from_traj, _make_obs_from_train_target

from moozi.replay import ReplayBuffer
from moozi.parameter_optimizer import ParameterServer
from moozi.rollout_worker import RolloutWorker
from moozi.laws import *
from moozi.planner import convert_tree_to_graph, make_gumbel_planner, make_planner
from moozi.nn import RootFeatures, TransitionFeatures

from lib import (
    training_suite_factory,
    make_test_worker_universe,
    make_reanalyze_universe,
    make_env_worker_universe,
    config,
    model,
    scalar_transform,
    Universe
)

# %%
vec_env = make_vec_env("MinAtar:SpaceInvaders-v1", num_envs=1)

# %%
config = OmegaConf.load(Path(__file__).parent / "config.yml")


jax.config.update("jax_disable_jit", True)
config.debug = True
config.env_worker.num_workers = 1
config.env_worker.num_envs = 1
OmegaConf.resolve(config)
print(OmegaConf.to_yaml(config, resolve=True))
OmegaConf.resolve(config)

# %%
ps = ParameterServer(training_suite_factory=training_suite_factory(config))
rb = ReplayBuffer(**config.replay)
vis = MinAtarVisualizer()

# %%
weights_path = "/home/zeyi/miniconda3/envs/moozi/.guild/runs/9adbbad8a45e4103ab71f98088b1d846/checkpoints/2670.pkl"
ps.restore(weights_path)

# %% 
def make_test_worker_universe(config):
    vec_env = make_vec_env(config.env.name, 1)
    obs_processor = make_obs_processor(
        num_rows=config.env.num_rows,
        num_cols=config.env.num_cols,
        num_channels=config.env.num_channels,
        num_stacked_frames=config.num_stacked_frames,
        dim_action=config.dim_action,
    ).vmap(batch_size=1)

    planner = make_planner(model=model, **config.test_worker.planner)

    final_law = sequential(
        [
            vec_env,
            obs_processor,
            planner,
            make_min_atar_gif_recorder(n_channels=6, root_dir="test_worker_gifs"),
            # make_traj_writer(1),
            make_reward_terminator(30),
        ]
    )
    tape = make_tape(seed=config.seed)
    tape.update(final_law.malloc())
    return Universe(tape, final_law)

# %%
u = make_test_worker_universe(config)
u.tape["params"] = ps.get_params()
u.tape["state"] = ps.get_state()

# %%
show_vis = True
for i in range(10):
    print(i)
    u.tick()

    if show_vis:
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
        # graph = convert_tree_to_graph(u.tape["tree"])
        # graph.draw(f"/tmp/graph_{i}.dot", prog="dot")

# %%
u.tape.keys()

# %%
traj = rb.get_trajs_batch(1)
# %%
traj = traj[0]

# %%
targets = []
traj_len = traj.action.shape[0]
for i in range(traj_len):
    target = make_target_from_traj(
        traj,
        start_idx=i,
        discount=rb.discount,
        num_unroll_steps=rb.num_unroll_steps,
        num_td_steps=rb.num_td_steps,
        num_stacked_frames=rb.num_stacked_frames,
    )
    value_diff = np.abs(target.n_step_return[0] - target.root_value[0])
    targets.append(target)

# %%
t = targets[-1]

# %%
