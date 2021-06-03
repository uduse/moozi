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
from acme.environment_loops.open_spiel_environment_loop import OpenSpielEnvironmentLoop
from acme.agents import agent as acme_agent
from acme.agents import replay as acme_replay

# %%
use_jit = True
if use_jit:
    jax.config.update("jax_disable_jit", not use_jit)

# %%
platform = "cpu"
jax.config.update("jax_platform_name", platform)

# In[2]:
seed = 0
key = jax.random.PRNGKey(seed)

# In[3]:
raw_env = open_spiel.python.rl_environment.Environment("catch(columns=7,rows=5)")
env = acme.wrappers.open_spiel_wrapper.OpenSpielWrapper(raw_env)
env = acme.wrappers.SinglePrecisionWrapper(env)
env_spec = acme.specs.make_environment_spec(env)
max_game_length = env.environment.environment.game.max_game_length()
dim_action = env_spec.actions.num_values
dim_image = env_spec.observations.observation.shape[0]
dim_repr = 1
print(env_spec)


# In[4]:
nn_spec = mz.nn.NeuralNetworkSpec(
    dim_image=dim_image, dim_repr=dim_repr, dim_action=dim_action
)
network = mz.nn.get_network(nn_spec)
learning_rate = 5e-3
optimizer = optax.adam(learning_rate)
print(nn_spec)


# %%
n_steps = 5
batch_size = 128
reverb_replay = acme_replay.make_reverb_prioritized_nstep_replay(
    env_spec, batch_size=batch_size, n_step=n_steps, max_replay_size=2000, discount=1.0
)


# %%
time_delta = 1.0
key, new_key = jax.random.split(key)
learner = mz.learner.SGDLearner(
    network=network,
    loss_fn=mz.loss.NStepPriorVanillaPolicyGradientLoss(),
    optimizer=optimizer,
    data_iterator=reverb_replay.data_iterator,
    random_key=new_key,
    loggers=[
        acme.utils.loggers.TerminalLogger(time_delta=time_delta, print_fn=print),
        mz.logging.JAXBoardLogger("learner", time_delta=time_delta),
    ],
)


# In[7]:
key, new_key = jax.random.split(key)
variable_client = acme.jax.variable_utils.VariableClient(learner, "")


# In[8]:
key, new_key = jax.random.split(key)
actor = mz.actor.PriorPolicyActor(
    environment_spec=env_spec,
    network=network,
    adder=reverb_replay.adder,
    variable_client=variable_client,
    random_key=new_key,
    loggers=[
        acme.utils.loggers.TerminalLogger(time_delta=time_delta, print_fn=print),
        mz.logging.JAXBoardLogger("actor", time_delta=time_delta),
    ],
    # epsilon=0.1,
    # temperature=1,
)

# %%
agent = acme_agent.Agent(
    actor=actor, learner=learner, min_observations=100, observations_per_step=100
)

# %%
# loop = OpenSpielEnvironmentLoop(environment=env, actors=[agent])
loop = OpenSpielEnvironmentLoop(environment=env, actors=[agent])

# %%
loop.run_episode()

# %%
num_steps = 2000
loop.run(num_steps=num_steps)

# %%
actor.close()
learner.close()
