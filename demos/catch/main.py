# %%
import functools
import os
import pickle
import random
import typing
from pathlib import Path
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
import tqdm.notebook
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
from IPython.display import Image, display
from moozi.actors.evaluator import Evaluator
from moozi.nn import NeuralNetwork, NNOutput
from moozi.policies.policy import PolicyFeed, PolicyResult
from moozi.replay import TrajectorySample, make_replay
from nptyping import NDArray
from reverb import rate_limiters
from reverb.trajectory_writer import TrajectoryColumn

# %%
use_jit = True
if use_jit:
    jax.config.update("jax_disable_jit", not use_jit)

platform = "gpu"
jax.config.update("jax_platform_name", platform)

print(jax.devices())

# %%
# env_columns, env_rows = 3, 3
# env_columns, env_rows = 5, 5
env_columns, env_rows = 6, 6
raw_env = open_spiel.python.rl_environment.Environment(
    f"catch(columns={env_columns},rows={env_rows})"
)
env = acme.wrappers.open_spiel_wrapper.OpenSpielWrapper(raw_env)
env = acme.wrappers.SinglePrecisionWrapper(env)
env_spec = acme.specs.make_environment_spec(env)

# %%
seed = 0
master_key = jax.random.PRNGKey(seed)

max_replay_size = 100000
max_episode_length = env.environment.environment.game.max_game_length()
num_unroll_steps = 3
num_stacked_frames = 1
num_td_steps = 100
batch_size = 256
discount = 0.99


# %%
dim_repr = 64
dim_action = env_spec.actions.num_values
frame_shape = env_spec.observations.observation.shape
stacked_frame_shape = (num_stacked_frames,) + frame_shape
nn_spec = mz.nn.NeuralNetworkSpec(
    stacked_frames_shape=stacked_frame_shape,
    dim_repr=dim_repr,
    dim_action=dim_action,
    repr_net_sizes=(128, 128),
    pred_net_sizes=(128, 128),
    dyna_net_sizes=(128, 128),
)
network = mz.nn.get_network(nn_spec)
lr = 2e-3
optimizer = optax.adam(lr)
print(nn_spec)


def convert_timestep(timestep):
    return timestep._replace(observation=timestep.observation[0])


def frame_to_str_gen(frame):
    for irow, row in enumerate(frame):
        for val in row:
            if np.isclose(val, 0.0):
                yield "."
                continue
            assert np.isclose(val, 1), val
            if irow == len(frame) - 1:
                yield "X"
            else:
                yield "O"
        yield "\n"


def frame_to_str(frame):
    return "".join(frame_to_str_gen(frame))


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
num_simulations = 10
mcts = mz.policies.MonteCarloTreeSearch(
    network=network,
    num_simulations=num_simulations,
    discount=0.99,
    dim_action=dim_action,
)


def train_policy_fn(params, feed: PolicyFeed) -> PolicyResult:
    mcts_tree = mcts(params, feed)

    action_probs = np.zeros((dim_action,), dtype=np.float32)
    for a, visit_count in mcts_tree.get_children_visit_counts().items():
        action_probs[a] = visit_count
    action_probs /= np.sum(action_probs)

    action, _ = mcts_tree.select_child()
    return PolicyResult(
        action=np.array(action, dtype=np.int32),
        extras={"tree": mcts_tree, "action_probs": action_probs},
    )


def eval_policy_fn(params, feed: PolicyFeed) -> PolicyResult:
    policy_result = train_policy_fn(params, feed)

    mcts_tree = policy_result.extras["tree"]

    child_values = np.zeros((dim_action,), dtype=np.float32)
    for action, value in mcts_tree.get_children_values().items():
        child_values[action] = value

    policy_result.extras.update(
        {
            "child_values": child_values,
        }
    )
    return policy_result


# %%
actor = mz.PlayerShell(
    env_spec,
    variable_client,
    train_policy_fn,
    reverb_replay.adder,
    new_key,
    num_stacked_frames=num_stacked_frames,
    loggers=[
        mz.logging.JAXBoardLogger("actor", time_delta=5.0),
        acme.utils.loggers.TerminalLogger(time_delta=5.0, print_fn=print),
    ],
)


# %%
obs_ratio = 20
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

# %%
evaluator = mz.actors.Evaluator(
    variable_client,
    eval_policy_fn,
    random_key=master_key,
    dim_action=dim_action,
    num_stacked_frames=num_stacked_frames,
)

# %%
def report_evaluation(
    env, evaluator: mz.actors.Evaluator, path, return_samples=100, display_samples=3
):
    output_string = ""
    eval_loop = OpenSpielEnvironmentLoop(
        environment=env, actors=[evaluator], logger=acme.utils.loggers.NoOpLogger()
    )

    return_results = [eval_loop.run_episode() for _ in range(return_samples)]
    avr_episode_return = np.array(
        list(map(lambda x: x["episode_return"], return_results))
    ).mean()
    output_string += f"Average Episode Return: {avr_episode_return}\n"

    for episode in range(display_samples):
        evaluator.reset_memory()
        eval_loop.run_episode()
        # timestep = env.reset()
        # evaluator.observe_first(convert_timestep(timestep))
        # while not timestep.last():
        #     action = evaluator.select_action(timestep.observation[0])
        #     timestep = env.step([action])
        #     evaluator.observe(action, convert_timestep(timestep))

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        policy_results = evaluator.m["policy_results"].get()
        last_frames = evaluator.m["last_frames"].get()

        assert len(policy_results) + 1 == len(last_frames)
        for i in range(len(last_frames)):
            frame = last_frames[i].reshape((env_rows, env_columns))
            output_string += frame_to_str(frame)

            if i < len(policy_results):
                output_string += "action: " + str(policy_results[i].action) + "\n"

                action_probs = policy_results[i].extras["action_probs"]
                output_string += "action probs: " + str(action_probs) + "\n"

                child_values = policy_results[i].extras["child_values"]
                output_string += "child values: " + str(child_values) + "\n"

                output_string += "\n"

                tree = mz.policies.monte_carlo_tree_search.convert_to_anytree(
                    policy_results[i].extras["tree"]
                )
                mz.policies.monte_carlo_tree_search.anytree_to_png(
                    tree, path / f"{episode}_{i}.png"
                )

        output_string += "\n\n"

    with open(path / "output.txt", "w") as f:
        f.write(output_string)

    with open(path / "weights.pkl", "w") as f:
        params = evaluator._variable_client.params
        f.write(str(pickle.dumps(params)))


# %%
epochs = 10
episodes_per_epoch = 30
return_samples = 30
for epoch in tqdm.notebook.tqdm(range(epochs)):
    for _ in range(episodes_per_epoch):
        loop.run_episode()
    report_evaluation(
        env, evaluator, "evaluation/" + str(epoch), return_samples=return_samples
    )

# %%
actor.close()
learner.close()
