# %%
import random
import typing
from typing import NamedTuple, Optional

from nptyping import NDArray
import acme
import acme.jax.utils
import acme.jax.variable_utils
import acme.wrappers
import chex
import dm_env
import jax
import jax.numpy as jnp
import numpy as np
import open_spiel
import optax
import reverb
from reverb.trajectory_writer import TrajectoryColumn
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
from reverb import rate_limiters
import moozi as mz

# %%
use_jit = True
if use_jit:
    jax.config.update("jax_disable_jit", not use_jit)

platform = "cpu"
jax.config.update("jax_platform_name", platform)

# %%
raw_env = open_spiel.python.rl_environment.Environment("catch(columns=7,rows=5)")
env = acme.wrappers.open_spiel_wrapper.OpenSpielWrapper(raw_env)
env = acme.wrappers.SinglePrecisionWrapper(env)
env_spec = acme.specs.make_environment_spec(env)
replay = acme_replay.make_reverb_prioritized_nstep_replay(env_spec)

# %%
max_replay_size = 1000
max_episode_length = 5
num_unroll_steps = 2
num_stacked_frames = 2
num_td_steps = 4
signature = mz.replay.make_signature(env_spec, max_episode_length)
print("signature:")
signature




# %%
replay_table = reverb.Table(
    name=DEFAULT_PRIORITY_TABLE,
    sampler=reverb.selectors.Uniform(),
    remover=reverb.selectors.Fifo(),
    max_size=max_replay_size,
    rate_limiter=reverb.rate_limiters.MinSize(1),
    signature=signature,
)
server = reverb.Server([replay_table], port=None)
address = f"localhost:{server.port}"
client = reverb.Client(address)

# %%
adder = mz.replay.MooZiAdder(
    client,
    num_unroll_steps=num_unroll_steps,
    num_stacked_frames=num_stacked_frames,
    num_td_steps=num_td_steps,
)

# %%
import copy

timestep = env.reset()
obs = mz.replay.Observation.from_env_timestep(timestep)
adder.add_first(obs)
while not timestep.last():
    action = random.choice(list(range(env.action_spec().num_values)))
    timestep = env.step([action])
    obs = mz.replay.Observation.from_env_timestep(timestep)
    ref = mz.replay.Reflection(
        action=np.int32(0),
        root_value=np.float32(1),
        child_visits=np.array([0.2, 0.5, 0.3], dtype=np.float32),
    )
    adder.add(ref, obs)

# %%
data_iterator = acme_datasets.make_reverb_dataset(
    table=DEFAULT_PRIORITY_TABLE,
    server_address=address,
    batch_size=1,
    # prefetch_size=1,
).as_numpy_iterator()

# %%
data = next(data_iterator).data
tree.map_structure(np.shape, data)
# %%
data = tree.map_structure(np.squeeze, data)
tree.map_structure(lambda x: x.dtype, data)
# %%
from acme.utils import tree_utils


# %%
sample = ReplaySample(**data)
sample
# %%
# def make_episodic_replay(
#     env_spec: acme.specs.EnvironmentSpec,
#     replay_table_name: str = DEFAULT_PRIORITY_TABLE,
#     port: Optional[str] = None,
#     max_replay_size: int = 10000,
#     num_unroll_steps: int = 5,
#     num_stacked_frames: int = 8,
#     num_td_steps: int = 20,
# ):

#     extras_spec = MooZiAdderExtra(
#         root_value=acme_specs.Array(shape=tuple(), dtype=float),
#         child_visits=acme_specs.BoundedArray(
#             shape=(env_spec.actions.num_values,), dtype=float, minimum=0, maximum=1
#         ),
#     )
#     signature = EpisodeAdder.signature(env_spec, extras_spec=extras_spec)
#     print(signature)
#     replay_table = reverb.Table(
#         name=replay_table_name,
#         sampler=reverb.selectors.Uniform(),
#         remover=reverb.selectors.Fifo(),
#         max_size=max_replay_size,
#         rate_limiter=reverb.rate_limiters.MinSize(1),
#         signature=signature,
#     )
#     server = reverb.Server([replay_table], port=port)

#     # The adder is used to insert observations into replay.
#     address = f"localhost:{server.port}"
#     client = reverb.Client(address)
#     adder = EpisodeAdder(client, max_sequence_length=1000)

#     # The dataset provides an interface to sample from replay.
#     data_iterator = acme_datasets.make_reverb_dataset(
#         table=replay_table_name,
#         server_address=address,
#         batch_size=batch_size,
#         prefetch_size=prefetch_size,
#     ).as_numpy_iterator()
#     return acme_replay.ReverbReplay(server, adder, data_iterator, client=client)


# # %%
# np.array([0, 0, 0, 1]).argmax()

# # %%
# replay = make_episodic_replay(
#     env_spec=env_spec,
# )

# server = replay.server
# adder = replay.adder

# # %%

# timestep = env.reset()
# adder.add_first(timestep)
# print(timestep)
# while not timestep.last():
#     action = random.randrange(0, env_spec.actions.num_values)
#     timestep = env.step([action])
#     adder.add(action, timestep, MooZiAdderExtra(0, [0, 1, 0]))
#     print(timestep)
