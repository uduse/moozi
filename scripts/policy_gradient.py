# %%
import acme
import acme.jax.utils
import acme.jax.variable_utils
import acme.wrappers
import jax
import jax.numpy as jnp
import moozi as mz
import numpy as np
import open_spiel
import optax
from acme.agents import agent as acme_agent
from acme.agents import replay as acme_replay
from acme.environment_loops.open_spiel_environment_loop import OpenSpielEnvironmentLoop
from acme.utils.loggers import NoOpLogger

# %%
use_jit = True
if use_jit:
    jax.config.update("jax_disable_jit", not use_jit)

# %%
platform = "cpu"
jax.config.update("jax_platform_name", platform)

# %%
seed = 0
master_key = jax.random.PRNGKey(seed)

# %%
raw_env = open_spiel.python.rl_environment.Environment("catch(columns=5,rows=5)")
env = acme.wrappers.open_spiel_wrapper.OpenSpielWrapper(raw_env)
env = acme.wrappers.SinglePrecisionWrapper(env)
env_spec = acme.specs.make_environment_spec(env)
print(env_spec)

# %% 
max_episode_length = env.environment.environment.game.max_game_length()
dim_action = env_spec.actions.num_values
frame_shape = env_spec.observations.observation.shape
dim_repr = 8

# %% 
num_unroll_steps = 2
num_stacked_frames = 2
num_td_steps = 4

stacked_frame_shape = (num_stacked_frames,) + frame_shape

# %%
nn_spec = mz.nn.NeuralNetworkSpec(
    stacked_frames_shape=stacked_frame_shape,
    dim_repr=dim_repr,
    dim_action=dim_action,
    repr_net_sizes=(256, 256),
    pred_net_sizes=(256, 256),
    dyna_net_sizes=(256, 256),
)
network = mz.nn.get_network(nn_spec)
lr = 5e-4
optimizer = optax.adam(lr)
print(nn_spec)


# %%
batch_size = 200
max_replay_size = 100_000
reverb_replay = mz.replay.make_replay(
    env_spec, max_episode_length=max_episode_length, batch_size=batch_size
)

# %%
time_delta = 10.0
weight_decay = 1e-4
entropy_reg = 7e-1
master_key, new_key = jax.random.split(master_key)
learner = mz.learner.SGDLearner(
    network=network,
    loss_fn=mz.loss.OneStepAdvantagePolicyGradientLoss(
        weight_decay=weight_decay, entropy_reg=entropy_reg
    ),
    optimizer=optimizer,
    data_iterator=reverb_replay.data_iterator,
    random_key=new_key,
    loggers=[
        acme.utils.loggers.TerminalLogger(time_delta=time_delta, print_fn=print),
        mz.logging.JAXBoardLogger("learner", time_delta=time_delta),
    ],
)


# %%
master_key, new_key = jax.random.split(master_key)
variable_client = acme.jax.variable_utils.VariableClient(learner, "")

# %%
master_key, new_key = jax.random.split(master_key)
policy = mz.policies.PriorPolicy(network, variable_client)
actor = mz.Actor(
    env_spec=env_spec,
    policy=policy,
    adder=reverb_replay.adder,
    random_key=new_key,
)

# %%
obs_ratio = 500
min_observations = 1000
agent = acme_agent.Agent(
    actor=actor,
    learner=learner,
    min_observations=min_observations,
    observations_per_step=int(obs_ratio),
)

# %%
loop = OpenSpielEnvironmentLoop(environment=env, actors=[agent], logger=NoOpLogger())
loop.run_episode()

# %%
num_steps = 1000
loop.run(num_steps=num_steps)

# %%
# manually close the loggers to avoid writing problems
actor.close()
learner.close()
