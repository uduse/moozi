# %%
import jax
import haiku as hk
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
from moozi.planner import convert_tree_to_graph
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
vec_env = make_vec_env("MinAtar:Breakout-v1", num_envs=1)

# %%
config = OmegaConf.load(Path(__file__).parent / "config.yml")
# config.debug = True
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
# weights_path = "/home/zeyi/miniconda3/envs/moozi/.guild/runs/561ace079fe84e9a9dc39944138f05f3/checkpoints/5900.pkl"
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

# %%
targets = [
    make_target_from_traj(
        traj,
        i,
        discount=config.discount,
        num_unroll_steps=config.num_unroll_steps,
        num_td_steps=config.num_td_steps,
        num_stacked_frames=config.num_stacked_frames,
    )
    for i in range(traj.frame.shape[0])
]


# %%
obs_from_target = _make_obs_from_train_target(
    stack_sequence_fields(targets),
    0,
    num_stacked_frames=config.num_stacked_frames,
    num_unroll_steps=config.num_unroll_steps,
    dim_action=config.dim_action,
)

# %%
rb.restore("/home/zeyi/moozi/examples/minatar_space_invaders/replay/415116.pkl")
rb.get_stats()
# %%
targets = rb.get_train_targets_batch(10)
targets = unstack_sequence_fields(targets, 10)

# %%
# generate traj
trajs = []
for i in range(1):
    trajs.extend(rollout_worker.run())
rb.add_trajs(trajs)
traj = trajs[0]

# %%
rea


# %%
for _ in range(1):
    print(ps.update(rb.get_train_targets_batch(batch_size=10), batch_size=10))

# %%
# images = []
# for i in range(traj.frame.shape[0]):
#     image = vis.make_image(traj.frame[i])
#     image = vis.add_descriptions(
#         image,
#         action=traj.action[i],
#         reward=traj.last_reward[i],
#     )
#     images.append(image)
# vis.cat_images(images)


# # %%
# reanalyze_worker.set("traj", trajs[0])
# # %%
# result = reanalyze_worker.run()

# # %%
# result

# # %%
# obs = _make_obs_from_train_target(
#     add_batch_dim(target),
#     step=0,
#     num_stacked_frames=config.num_stacked_frames,
#     num_unroll_steps=config.num_unroll_steps,
#     dim_action=config.dim_action,
# )
# # %%
# rb.add_trajs(trajs)

# # %%
# # %%
# for _ in range(100):
#     print(ps.update(batch, batch_size=config.train.batch_size))

# # %%
# import random

# targets = unstack_sequence_fields(batch, batch.frame.shape[0])
# target = random.choice(targets)
# target = add_batch_dim(target)

# # %%
# obs = _make_obs_from_train_target(
#     target, 0, config.num_stacked_frames, config.num_unroll_steps, config.dim_action
# )

# # %%
# images = []
# for i in range(target.frame.shape[1] - 1):
#     image = vis.make_image(target.frame[0, i])
#     image = vis.add_descriptions(image, action=target.action[0, i + 1])
#     images.append(image)
# display(vis.cat_images(images))


# # %%
# train_target = rb.get_train_targets_batch(10)
# reanalyze_worker.set("train_target", train_target)
# updated_target = reanalyze_worker.run()

# # %%
# tree = reanalyze_worker.universe.tape["tree"]

# # %%
# updated_target = stack_sequence_fields(updated_target)

# # %%
# from moozi.planner import convert_tree_to_graph

# graph = convert_tree_to_graph(tree)

# # %%
# graph.draw("/tmp/graph.dot", prog="dot")

# # %%
# updated_target.root_value

# # %%
# train_target.root_value

# # %%


# ## CHEATSHEET

# visualize targets
target = targets[0]
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

# %%
image = vis.make_image(u.tape["frame"][0])
image = vis.add_descriptions(
    image,
    action=u.tape["action"][0],
    q_values=u.tape["q_values"][0],
    action_probs=u.tape["action_probs"][0],
    prior_probs=u.tape["prior_probs"][0],
    reward=u.tape["reward"][0],
    visit_counts=u.tape["visit_counts"][0],
)
display(image)
# graph = convert_tree_to_graph(tape["tree"])
# graph.draw(f"/tmp/graph_{i}.dot", prog="dot")

# %%
# nn_out, _ = model.root_inference(
#     u.tape["params"],
#     u.tape["state"],
#     RootFeatures(u.tape["obs"], np.array(0)),
#     False,
# )
# nn_out.value.item()

# %%
# nn_out, _ = model.root_inference(
#     u.tape["params"],
#     u.tape["state"],
#     RootFeatures(u.tape["obs"], np.array(0)),
#     True,
# )
# scalar_transform.inverse_transform(jax.nn.softmax(nn_out.value)).item()

# # %%
# last_hidden = nn_out.hidden_state[0, ...]
# for action in [3]:
#     print(f"{action=}")
#     nn_out1, _ = model.trans_inference_unbatched(
#         u.tape["params"],
#         u.tape["state"],
#         TransitionFeatures(last_hidden, np.array(action)),
#         is_training=True,
#     )
#     value_probs = jax.nn.softmax(nn_out1.value)
#     value = scalar_transform.inverse_transform(
#         value_probs.reshape((1, *value_probs.shape))
#     )
#     reward_probs = jax.nn.softmax(nn_out1.reward)
#     reward = scalar_transform.inverse_transform(
#         reward_probs.reshape((1, *reward_probs.shape))
#     )
#     print(f"{value_probs.tolist()=}")
#     print(f"{value.tolist()=}")
#     print(f"{reward_probs.tolist()=}")
#     print(f"{reward.tolist()=}")
#     print()

# nn_out2, _ = model.trans_inference_unbatched(
#     u.tape["params"],
#     u.tape["state"],
#     TransitionFeatures(last_hidden, np.array(action)),
#     is_training=False,
# )
# print(f"{nn_out2.value.tolist()=}")
# print(f"{nn_out2.reward.tolist()=}")
# print("\n")

# %% 

pb_c_init = 1.25
pb_c_base = 19000
visit_counts = np.array([20, 0])
node_visit = sum(visit_counts)
prior_probs = np.array([0.9, 0.1])
pb_c = pb_c_init + jnp.log((node_visit + pb_c_base + 1.) / pb_c_base)
policy_score = jnp.sqrt(node_visit) * pb_c * prior_probs / (visit_counts + 1)
print(policy_score)
# print(policy_score[0] / policy_score[1])
# %%