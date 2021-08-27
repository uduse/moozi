# %%
import functools
import pickle
import random
import typing
from typing import NamedTuple, Optional

import acme
import acme.jax.utils
import acme.wrappers
import anytree
import chex
import dm_env
import jax
import jax.numpy as jnp
import moozi as mz
import numpy as np
import open_spiel
import optax
import reverb
import rlax
import tree
from absl.testing import absltest, parameterized
from acme import datasets as acme_datasets
from acme import specs as acme_specs
from acme.adders.reverb import DEFAULT_PRIORITY_TABLE, EpisodeAdder, episode
from acme.adders.reverb import test_utils as acme_test_utils
from acme.adders.reverb.base import ReverbAdder, Trajectory
from acme.agents import agent as acme_agent
from acme.agents import replay as acme_replay
from acme.environment_loops.open_spiel_environment_loop import \
    OpenSpielEnvironmentLoop
from acme.jax.variable_utils import VariableClient
from acme.utils import tree_utils
from acme.utils.loggers.base import NoOpLogger
from moozi.nn import NeuralNetwork, NeuralNetworkOutput
from moozi.policies.policy import PolicyFeed, PolicyResult
from moozi.replay import Trajectory, make_replay
from nptyping import NDArray
from reverb import rate_limiters
from reverb.trajectory_writer import TrajectoryColumn

# %%
from core import *

# %%
reverb_replay = make_replay(
    env_spec,
    max_episode_length=max_episode_length,
    batch_size=batch_size,
    max_replay_size=max_replay_size,
)

# %%
master_key, new_key = jax.random.split(master_key)
# params = network.init(new_key)

# %%
# master_key, new_key = jax.random.split(master_key)
data_iterator = mz.replay.post_process_data_iterator(
    reverb_replay.data_iterator,
    batch_size,
    discount,
    num_unroll_steps,
    num_td_steps,
    num_stacked_frames,
)

# %%
weight_decay = 1e-4
# entropy_reg = 0.5
loss_fn = mz.loss.MuZeroLoss(num_unroll_steps, weight_decay)
learner = mz.learner.SGDLearner(
    network,
    loss_fn=loss_fn,
    optimizer=optimizer,
    data_iterator=data_iterator,
    random_key=new_key,
    loggers=[
        mz.logging.JAXBoardLogger("learner", time_delta=5.0),
        acme.utils.loggers.TerminalLogger(time_delta=5.0, print_fn=print),
    ],
)
variable_client = VariableClient(learner, None)

# %%
master_key, new_key = jax.random.split(master_key)
single_roll_monte_carlo_fn = mz.policies.make_single_roll_monte_carlo_fn(
    network, num_unroll_steps
)
policy_epsilon = 0.1

def policy_fn(params, feed: PolicyFeed) -> PolicyResult:
    mc_result = single_roll_monte_carlo_fn(params, feed)
    action_probs = rlax.epsilon_greedy(policy_epsilon).probs(
        mc_result.actions_reward_sum
    )
    legal_action_probs = action_probs * feed.legal_actions_mask
    action = rlax.categorical_sample(feed.random_key, legal_action_probs)
    return PolicyResult(
        action=action,
        extras={"tree": mc_result.tree, "action_probs": legal_action_probs},
    )

# %% 
actor = mz.MuZeroActor(
    env_spec,
    variable_client,
    policy_fn,
    reverb_replay.adder,
    new_key,
    num_stacked_frames=num_stacked_frames,
    loggers=[
        mz.logging.JAXBoardLogger("actor", time_delta=5.0),
        acme.utils.loggers.TerminalLogger(time_delta=5.0, print_fn=print),
    ],
)

# %%
obs_ratio = 10
min_observations = 0
agent = acme_agent.Agent(
    actor=actor,
    learner=learner,
    min_observations=min_observations,
    observations_per_step=int(obs_ratio),
)

# %%
loop = OpenSpielEnvironmentLoop(
    environment=env,
    actors=[agent],
    logger=acme.utils.loggers.TerminalLogger(time_delta=5.0, print_fn=print),
)
loop.run_episode()

# %%
# num_episodes = 1000
num_episodes = 100000
eval_frequency = 1000
for i in range(num_episodes):
    loop.run_episode()
    if i % eval_frequency == 0:
        with open(f"{i}.pickle", "wb") as f:
            pickle.dump(variable_client.params, f)

# %%
with open("./pickles/99000.pickle", "rb") as f:
    params = pickle.load(f)
print(list(params.keys()))

# %%
class Node(NamedTuple):
    network_output: NeuralNetworkOutput
    prior: jnp.ndarray
    children: list


def _expand_node(network: NeuralNetwork, params, parent: Node, action_space_size: int):
    children = []
    child_probs = rlax.safe_epsilon_softmax(1e-5, 1).probs(
        parent.network_output.policy_logits
    )
    for action_idx in range(action_space_size):
        child_network_output = network.recurrent_inference_unbatched(
            params,
            parent.network_output.hidden_state,
            action_idx,
        )
        child_prob = child_probs[action_idx]
        child_node = Node(
            network_output=child_network_output,
            prior=child_prob,
            children=[],
        )
        parent.children.append((action_idx, child_node))
        children.append(child_node)
    return children


def _build_tree(params, feed: PolicyFeed) -> Node:
    key = feed.random_key
    action_space_size = jnp.size(feed.legal_actions_mask)

    root_network_output = network.initial_inference_unbatched(
        params, feed.stacked_frames
    )

    root_node = Node(
        network_output=root_network_output,
        prior=jnp.array(0),
        children=[],
    )
    frontier = [root_node]

    for _ in range(num_unroll_steps):
        next_frontier = []
        for node in frontier:
            children = _expand_node(network, params, node, action_space_size)
            next_frontier.extend(children)
        frontier = next_frontier
    return root_node


# %%
# reverb_replay = make_replay(
#     env_spec, max_episode_length=max_episode_length, batch_size=batch_size
# )
# actor = mz.MuZeroActor(
#     env_spec,
#     policy,
#     reverb_replay.adder,
#     new_key,
#     num_stacked_frames=num_stacked_frames,
#     loggers=[
#         mz.logging.JAXBoardLogger("actor", time_delta=5.0),
#         acme.utils.loggers.TerminalLogger(time_delta=5.0, print_fn=print),
#     ],
# )

# actor.reset_memory()
# loop = OpenSpielEnvironmentLoop(environment=env, actors=[actor], logger=NoOpLogger())
# loop.run_episode()


# # %%
# x = next(reverb_replay.data_iterator)
# x = tree.map_structure(lambda x: x[0], x).data
# x

# # %%
# idx = 0
# policy_result_tree = actor.m["policy_results"].get()[idx].extras["tree"]
# anytree_root = mz.utils.convert_to_anytree(policy_result_tree)
# print(anytree.RenderTree(anytree_root))

# last_frame = actor.m["last_frames"].get()[idx]
# print(mz.utils.frame_to_str(last_frame.reshape(5, 5)))
# mz.utils.anytree_to_png(anytree_root, "./policy_tree.png")
# from IPython.display import Image

# Image("./policy_tree.png")

# # # %%
# # for i in range(40, 40 + 4):
# #     frame = actor.m["last_frames"].get()[i]
# #     frame = frame.reshape((3, 3)).tolist()
# #     print(mz.utils.frame_to_str(frame))
# #     if i < len(actor.m["policy_results"].get()):
# #         policy_result = actor.m["policy_results"].get()[i]
# #         probs = np.array(policy_result.extras["action_probs"])
# #         probs = np.round(probs, 2)
# #         print("action probs:".ljust(20), probs)
# #         print(
# #             "legal action probs:".ljust(20), np.round(policy_result.extras["legal_action_probs"], 2)
# #         )
# #         print(
# #             "actions reward sum:".ljust(20), np.round(policy_result.extras["actions_reward_sum"], 2)
# #         )
# #         policy_result.extras["action_probs"] = probs.tolist()
# #     print("\n")


# # %%
# jnp.save("./params.npy", params)

# # %%
# learner.close()
# actor.close()
