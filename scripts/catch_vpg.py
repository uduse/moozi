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
jax_platform_name = "cpu"
jax.config.update("jax_platform_name", jax_platform_name)

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
dim_repr = 3
print(env_spec)


# In[4]:
nn_spec = mz.nn.NeuralNetworkSpec(
    dim_image=dim_image, dim_repr=dim_repr, dim_action=dim_action
)
network = mz.nn.get_network(nn_spec)
learning_rate = 1e-3
optimizer = optax.adam(learning_rate)
print(nn_spec)


# In[5]:
batch_size = 64
n_steps = 5
reverb_replay = acme_replay.make_reverb_prioritized_nstep_replay(
    env_spec, batch_size=batch_size, n_step=n_steps
)


# In[6]:
use_log = False
loggers = [
    acme.utils.loggers.TerminalLogger(time_delta=5.0, print_fn=print),
]
if use_log:
    loggers.append(mz.logging.JAXBoardLogger("learner", time_delta=5.0))
learner = mz.learner.SGDLearner(
    network=network,
    loss_fn=mz.loss.NStepPriorVanillaPolicyGradientLoss(),
    optimizer=optimizer,
    data_iterator=reverb_replay.data_iterator,
    random_key=jax.random.PRNGKey(996),
    loggers=loggers,
)


# In[7]:
key, new_key = jax.random.split(key)
variable_client = acme.jax.variable_utils.VariableClient(learner, None)


# In[8]:
key, new_key = jax.random.split(key)
actor = mz.actor.PriorPolicyActor(
    environment_spec=env_spec,
    network=network,
    adder=reverb_replay.adder,
    variable_client=variable_client,
    random_key=new_key,
    #     epsilon=0.1,
    #     temperature=1
)


# %%
agent = acme_agent.Agent(
    actor=actor, learner=learner, min_observations=100, observations_per_step=1
)


# In[10]:
loop = OpenSpielEnvironmentLoop(environment=env, actors=[agent])

# %%
loop.run_episode()

# In[11]:
num_steps = 100
loop.run(num_steps=num_steps)

# %%
