# %%
from omegaconf import OmegaConf
import numpy as np
from IPython.display import display
from loguru import logger
import ray
from functools import partial
from moozi.laws import MinAtarVisualizer

from moozi.logging import JAXBoardLoggerRemote, TerminalLoggerRemote
from moozi.nn.training import make_target_from_traj
from moozi.replay import ReplayBuffer
from moozi.parameter_optimizer import ParameterServer
from moozi.rollout_worker import RolloutWorker
from moozi.laws import *
from moozi.planner import convert_tree_to_graph

from lib import (
    training_suite_factory,
    make_test_worker_universe,
    make_reanalyze_universe,
    make_env_worker_universe,
    config,
)

# %%
vec_env = make_vec_env("MinAtar:SpaceInvaders-v1", num_envs=2)

# %%
config = OmegaConf.load(Path(__file__).parent / "config.yml")
config.train.env_worker.num_workers = 1
config.train.env_worker.num_envs = 1
config.replay.min_size = 1
OmegaConf.resolve(config)
print(OmegaConf.to_yaml(config, resolve=True))
OmegaConf.resolve(config)

# %%
ps = ParameterServer(training_suite_factory=training_suite_factory(config))
rb = ReplayBuffer(**config.replay)
rb.min_size = 1
vis = MinAtarVisualizer()

# %%
# weights_path = "/home/zeyi/miniconda3/envs/moozi/.guild/runs/0ad7dfc32c9f4685a02ba66b3731ef12/checkpoints/7589.pkl"
# ps.restore(weights_path)

# %%
rollout_worker = RolloutWorker(
    partial(make_env_worker_universe, config), name=f"rollout_worker"
)
rollout_worker.set("params", ps.get_params())
rollout_worker.set("state", ps.get_state())

# %%
reanalyze_worker = RolloutWorker(
    partial(make_reanalyze_universe, config), name=f"reanalyze_worker"
)
reanalyze_worker.set("params", ps.get_params())
reanalyze_worker.set("state", ps.get_state())

# %%
for i in range(100):
    reanalyze_worker.universe.tick()
    tape = reanalyze_worker.universe.tape
    image = vis.make_image(tape["frame"][0])
    image = vis.add_descriptions(
        image,
        action=tape["action"][0],
        q_values=tape["q_values"][0],
        action_probs=tape["action_probs"][0],
        prior_probs=tape["prior_probs"][0],
        reward=tape["reward"][0],
        visit_counts=tape["visit_counts"][0],
    )
    display(image)
    # graph = convert_tree_to_graph(tape["tree"])
    # graph.draw(f"/tmp/graph_{i}.dot", prog="dot")

# %%
trajs = []
for i in range(1):
    trajs.extend(rollout_worker.run())
rb.add_trajs(trajs)

# %%
for _ in range(1000):
    print(ps.update(rb.get_train_targets_batch(batch_size=10), batch_size=10))

# %%
images = []
for i in range(traj.frame.shape[0]):
    image = vis.make_image(traj.frame[i])
    image = vis.add_descriptions(
        image,
        action=traj.action[i],
        reward=traj.last_reward[i],
    )
    images.append(image)
vis.cat_images(images)


# %%
reanalyze_worker.set('traj', trajs[0])
# %%
result = reanalyze_worker.run()

# %%
result

# %%
obs = _make_obs_from_train_target(
    add_batch_dim(target),
    step=0,
    num_stacked_frames=config.num_stacked_frames,
    num_unroll_steps=config.num_unroll_steps,
    dim_action=config.dim_action,
)
# %%
rb.add_trajs(trajs)

# %%
# %%
for _ in range(100):
    print(ps.update(batch, batch_size=config.train.batch_size))

# %%
import random

targets = unstack_sequence_fields(batch, batch.frame.shape[0])
target = random.choice(targets)
target = add_batch_dim(target)

# %%
obs = _make_obs_from_train_target(
    target, 0, config.num_stacked_frames, config.num_unroll_steps, config.dim_action
)

# %%
images = []
for i in range(target.frame.shape[1] - 1):
    image = vis.make_image(target.frame[0, i])
    image = vis.add_descriptions(image, action=target.action[0, i + 1])
    images.append(image)
display(vis.cat_images(images))


# %%
train_target = rb.get_train_targets_batch(10)
reanalyze_worker.set("train_target", train_target)
updated_target = reanalyze_worker.run()

# %%
tree = reanalyze_worker.universe.tape["tree"]

# %%
updated_target = stack_sequence_fields(updated_target)

# %%
from moozi.planner import convert_tree_to_graph

graph = convert_tree_to_graph(tree)

# %%
graph.draw("/tmp/graph.dot", prog="dot")

# %%
updated_target.root_value

# %%
train_target.root_value

# %%


## CHEATSHEET

# visualize targets
targets = [targets[0]]
for i, target in enumerate(targets):
    print(i)
    images = []
    for i in range(target.frame.shape[0]):
        image = vis.make_image(target.frame[i])
        descriptions = {}
        if i >= config.num_stacked_frames - 1:
            offset = i + 1 - config.num_stacked_frames
            descriptions["n_step_return"] = target.n_step_return[offset]
            descriptions["root_value"] = target.root_value[offset]
            descriptions["action_probs"] = target.action_probs[offset]
            descriptions["reward"] = target.last_reward[offset]
        if i < target.frame.shape[0] - 1:
            descriptions["action"] = target.action[i + 1]
        image = vis.add_descriptions(image, **descriptions)
        images.append(image)
    display(vis.cat_images(images))
