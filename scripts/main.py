# %%
import functools
import random
import typing
from typing import NamedTuple, Optional

import acme
import acme.jax.utils
import acme.wrappers
import chex
import dm_env
import jax
import jax.numpy as jnp
import moozi as mz
import numpy as np
import open_spiel
import optax
import reverb
import tree
from absl.testing import absltest, parameterized
from acme import datasets as acme_datasets
from acme import specs as acme_specs
from acme.adders.reverb import DEFAULT_PRIORITY_TABLE, EpisodeAdder
from acme.adders.reverb import test_utils as acme_test_utils
from acme.adders.reverb.base import ReverbAdder, Trajectory
from acme.agents import agent as acme_agent
from acme.agents import replay as acme_replay
from acme.environment_loops.open_spiel_environment_loop import OpenSpielEnvironmentLoop
from acme.jax.variable_utils import VariableClient
from acme.utils import tree_utils
from acme.utils.loggers.base import NoOpLogger
from moozi.replay import Trajectory, make_replay
from nptyping import NDArray
from reverb import rate_limiters
from reverb.trajectory_writer import TrajectoryColumn

# %%
use_jit = True
if use_jit:
    jax.config.update("jax_disable_jit", not use_jit)

platform = "gpu"
jax.config.update("jax_platform_name", platform)

# %%
raw_env = open_spiel.python.rl_environment.Environment("catch(columns=7,rows=5)")
env = acme.wrappers.open_spiel_wrapper.OpenSpielWrapper(raw_env)
env = acme.wrappers.SinglePrecisionWrapper(env)
env_spec = acme.specs.make_environment_spec(env)

# %%
seed = 0
master_key = jax.random.PRNGKey(seed)
max_replay_size = 1000000
max_episode_length = env.environment.environment.game.max_game_length()
num_unroll_steps = 3
num_stacked_frames = 2
num_td_steps = 100
batch_size = 256
discount = 0.99
dim_action = env_spec.actions.num_values
frame_shape = env_spec.observations.observation.shape

stacked_frame_shape = (num_stacked_frames,) + frame_shape

# %%
reverb_replay = make_replay(
    env_spec, max_episode_length=max_episode_length, batch_size=batch_size
)

# %%
dim_repr = 32
nn_spec = mz.nn.NeuralNetworkSpec(
    stacked_frames_shape=stacked_frame_shape,
    dim_repr=dim_repr,
    dim_action=dim_action,
    repr_net_sizes=(128, 128),
    pred_net_sizes=(128, 128),
    dyna_net_sizes=(128, 128),
)
network = mz.nn.get_network(nn_spec)
lr = 5e-4
optimizer = optax.adam(lr)
print(nn_spec)


# %%
master_key, new_key = jax.random.split(master_key)
params = network.init(new_key)

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
entropy_reg = 0.5
loss_fn = mz.loss.MCTSLoss(num_unroll_steps, weight_decay)
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
policy = mz.policies.SingleRollMonteCarlo(network, variable_client)
actor = mz.MuZeroActor(
    env_spec,
    policy,
    reverb_replay.adder,
    new_key,
    num_stacked_frames=num_stacked_frames,
    loggers=[
        mz.logging.JAXBoardLogger("actor", time_delta=5.0),
        acme.utils.loggers.TerminalLogger(time_delta=5.0, print_fn=print),
    ],
)

# %%
obs_ratio = 1000
min_observations = 0
agent = acme_agent.Agent(
    actor=actor,
    learner=learner,
    min_observations=min_observations,
    observations_per_step=int(obs_ratio),
)

# %%
num_episodes = 100000
loop = OpenSpielEnvironmentLoop(
    environment=env,
    actors=[agent],
    logger=acme.utils.loggers.TerminalLogger(time_delta=5.0, print_fn=print),
)
loop.run(num_episodes=num_episodes)

# %%
reverb_replay = make_replay(
    env_spec, max_episode_length=max_episode_length, batch_size=batch_size
)
actor = mz.MuZeroActor(
    env_spec,
    policy,
    reverb_replay.adder,
    new_key,
    num_stacked_frames=num_stacked_frames,
    loggers=[
        mz.logging.JAXBoardLogger("actor", time_delta=5.0),
        acme.utils.loggers.TerminalLogger(time_delta=5.0, print_fn=print),
    ],
)


def convert_timestep(timestep):
    return timestep._replace(observation=timestep.observation[0])


# %%
actor.reset_memory()
loop = OpenSpielEnvironmentLoop(environment=env, actors=[actor], logger=NoOpLogger())
loop.run_episode()

# %%
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
import anytree
import uuid


def get_uuid():
    return uuid.uuid4().hex[:8]


# %%
def convert_to_anytree(policy_tree_root, anytree_root=None, action="_"):
    anytree_child = anytree.Node(
        id=get_uuid(),
        name=action,
        parent=anytree_root,
        prior=policy_tree_root.prior,
        reward=np.round(np.array(policy_tree_root.network_output.reward).item(), 3),
        value=np.round(np.array(policy_tree_root.network_output.value).item(), 3),
    )
    for next_action, policy_tree_child in policy_tree_root.children:
        convert_to_anytree(policy_tree_child, anytree_child, next_action)
    return anytree_child


# %%
policy_result_tree = actor.m["policy_results"].get()[0].extras["tree"]
anytree_root = convert_to_anytree(policy_result_tree)
print(anytree.RenderTree(anytree_root))


# %%
from anytree.exporter import DotExporter, UniqueDotExporter


def nodeattrfunc(node):
    return (
        f'"reward: {node.reward:.3f}\nvalue: {node.value:.3f}"'
    )
    
def edgeattrfunc(parent, child):
    return f"label=\"{child.name} ({child.prior:.3f})\""

ex = UniqueDotExporter(
    anytree_root,
    nodenamefunc=lambda node: node.id,
    nodeattrfunc=lambda node: f"label={nodeattrfunc(node)}",
    edgeattrfunc=edgeattrfunc,
)
ex.to_picture("/tmp/policy_tree.png")
from IPython.display import Image

Image("/tmp/policy_tree.png")


# %%
for i in range(len(actor.m["last_frames"])):
    frame = actor.m["last_frames"].get()[i]
    frame = frame.reshape((5, 7)).tolist()
    print(frame_to_str(frame))
    if i < len(actor.m["policy_results"].get()):
        policy_result = actor.m["policy_results"].get()[i]
        probs = np.array(policy_result.extras["action_probs"])
        probs = np.round(probs, 2)
        print("action probs:".ljust(20), probs)
        print(
            "legal action probs:".ljust(20), policy_result.extras["legal_action_probs"]
        )
        print(
            "actions reward sum:".ljust(20), policy_result.extras["actions_reward_sum"]
        )
        policy_result.extras["action_probs"] = probs.tolist()
    print("\n")

# %%
learner.close()
actor.close()
