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
from acme.environment_loops.open_spiel_environment_loop import OpenSpielEnvironmentLoop
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

# %%
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
mcts = mz.policies.MonteCarloTreeSearch(
    network=network, num_simulations=50, discount=0.99, dim_action=dim_action
)


def policy_fn(params, feed: PolicyFeed) -> PolicyResult:
    mcts_result = mcts(params, feed)
    action, _ = mcts_result.tree.select_child()
    action_probs = np.zeros((dim_action,), dtype=np.float32)
    for action, visit_count in mcts_result.visit_counts.items():
        action_probs[action] = visit_count
    action_probs /= np.sum(action_probs)
    return PolicyResult(
        action=action,
        extras={"tree": mcts_result.tree, "action_probs": action_probs},
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
obs_ratio = 50
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

# # %%
# # num_episodes = 1000
# num_episodes = 100000
# eval_frequency = 1000
# for i in range(num_episodes):
#     loop.run_episode()
#     if i % eval_frequency == 0:
#         with open(f"{i}.pickle", "wb") as f:
#             pickle.dump(variable_client.params, f)
