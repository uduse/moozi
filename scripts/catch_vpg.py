#!/usr/bin/env python
# coding: utf-8

# In[2]:


import random
import copy
from collections import namedtuple
from dataclasses import dataclass
import datetime
import typing
import functools
from pprint import pprint

import jax
import jax.numpy as jnp
from jax import grad, value_and_grad, jit, vmap
from jax.experimental import optimizers
from jax.experimental import stax
import optax
import haiku as hk
from jax.tree_util import tree_flatten
import jaxboard

import pyspiel
import open_spiel
import dm_env
import acme
import acme.wrappers
import acme.jax.utils
import acme.jax.variable_utils
from acme.agents import agent as acme_agent
from acme.agents import replay as acme_replay
from acme.environment_loops.open_spiel_environment_loop import OpenSpielEnvironmentLoop
from acme.wrappers.open_spiel_wrapper import OpenSpielWrapper

from tqdm.notebook import tqdm
import numpy as np
import trueskill

import moozi as mz


# In[2]:


seed = 0
key = jax.random.PRNGKey(seed)


# In[3]:


raw_env = open_spiel.python.rl_environment.Environment('catch(columns=7,rows=5)')
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
    dim_image=dim_image,
    dim_repr=dim_repr,
    dim_action=dim_action
)
network = mz.nn.get_network(nn_spec)
learning_rate = 1e-4
optimizer = optax.adam(learning_rate)
print(nn_spec)


# In[5]:


batch_size = 32
n_steps=5
reverb_replay = acme_replay.make_reverb_prioritized_nstep_replay(
    env_spec, batch_size=batch_size, n_step=n_steps)


# In[6]:


learner = mz.learner.MooZiLearner(
    network=network,
    loss_fn=mz.loss.n_step_prior_vanilla_policy_gradient_loss,
    optimizer=optimizer,
    data_iterator=reverb_replay.data_iterator,
    random_key=jax.random.PRNGKey(996),
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


# In[9]:


agent = acme_agent.Agent(
    actor=actor,
    learner=learner,
    min_observations=100,
    observations_per_step=1
)


# In[10]:


loop = OpenSpielEnvironmentLoop(environment=env, actors=[agent])


# In[11]:


num_steps = 5_000
loop.run(num_steps=num_steps)


# In[12]:


print(learner._jaxboard_logger._log_dir)


# In[19]:


logits = -np.ones(30)
logits[0] = 1
jax.nn.softmax(logits)


# In[ ]:























