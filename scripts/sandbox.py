# %%
import random
import typing
from typing import Optional

import acme
import acme.jax.utils
import acme.jax.variable_utils
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
from pandas.core.computation.ops import Op
from reverb import rate_limiters

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



# %%
def make_episodic_replay(
    env_spec: acme.specs.EnvironmentSpec,
    replay_table_name: str = DEFAULT_PRIORITY_TABLE,
    port: Optional[str] = None,
    max_replay_size: int = 10000,
    batch_size: int = 2048,
    prefetch_size: int = 4,
    num_unroll_steps: int = 5,
    num_stacked_frames: int = 8,
    num_td_steps: int = 20,
):

    extras_spec = MooZiAdderExtra(
        root_value=acme_specs.Array(shape=tuple(), dtype=float),
        child_visits=acme_specs.BoundedArray(
            shape=(env_spec.actions.num_values,), dtype=float, minimum=0, maximum=1
        ),
    )
    signature = EpisodeAdder.signature(env_spec, extras_spec=extras_spec)
    print(signature)
    replay_table = reverb.Table(
        name=replay_table_name,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        max_size=max_replay_size,
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature=signature,
    )
    server = reverb.Server([replay_table], port=port)

    # The adder is used to insert observations into replay.
    address = f"localhost:{server.port}"
    client = reverb.Client(address)
    adder = EpisodeAdder(client, max_sequence_length=1000)

    # The dataset provides an interface to sample from replay.
    data_iterator = acme_datasets.make_reverb_dataset(
        table=replay_table_name,
        server_address=address,
        batch_size=batch_size,
        prefetch_size=prefetch_size,
    ).as_numpy_iterator()
    return acme_replay.ReverbReplay(server, adder, data_iterator, client=client)


# %%
replay = make_episodic_replay(
    env_spec=env_spec,
)

server = replay.server
adder = replay.adder

# %%

timestep = env.reset()
adder.add_first(timestep)
print(timestep)
while not timestep.last():
    action = random.randrange(0, env_spec.actions.num_values)
    timestep = env.step([action])
    adder.add(action, timestep, MooZiAdderExtra(0, [0, 1, 0]))
    print(timestep)
