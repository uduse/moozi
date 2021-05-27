#!/usr/bin/env python
# coding: utf-8

# In[8]:


import random
import copy
from collections import namedtuple
from dataclasses import dataclass
import datetime
import typing
import functools
from pprint import pprint


# In[9]:


import jax
import jax.numpy as jnp
from jax import grad, value_and_grad, jit, vmap
from jax.experimental import optimizers
from jax.experimental import stax
import optax
import haiku as hk
from jax.tree_util import tree_flatten

import pyspiel
import open_spiel
import dm_env
import acme
import acme.wrappers
import acme.jax.utils
from acme.agents import agent as acme_agent
from acme.agents import replay as acme_replay
from acme.environment_loops.open_spiel_environment_loop import OpenSpielEnvironmentLoop
from acme.wrappers.open_spiel_wrapper import OpenSpielWrapper

from tqdm.notebook import tqdm
import numpy as np
import trueskill


# In[ ]:


import moozi as mz


# In[ ]:


# %run hardware_sanity_check.ipynb


# In[ ]:


# %run moozi/utils.py


# In[ ]:


# OpenSpiel environment, not using it for now since not supported by the latest relased Acme
# raw_env = open_spiel.python.rl_environment.Environment('catch(columns=8,rows=4)')
raw_env = open_spiel.python.rl_environment.Environment('catch(columns=7,rows=5)')
env = acme.wrappers.open_spiel_wrapper.OpenSpielWrapper(raw_env)
env = acme.wrappers.SinglePrecisionWrapper(env)
env_spec = acme.specs.make_environment_spec(env)
max_game_length = env.environment.environment.game.max_game_length()
dim_action = env_spec.actions.num_values
dim_image = env_spec.observations.observation.shape
dim_repr = 3
print(env_spec)
# print_traj_in_env(env)


# In[ ]:


nn_spec = mz.nn.NeuralNetworkSpec(
    dim_image=dim_image,
    dim_repr=dim_repr,
    dim_action=dim_action
)
print(nn_spec)
network = mz.nn.get_network(nn_spec)
optimizer = optax.adam(1e-4)


# In[ ]:


batch_size = 16
n_steps=5
reverb_replay = acme_replay.make_reverb_prioritized_nstep_replay(
    env_spec, batch_size=batch_size, n_step=n_steps)
# reverb_replay = acme_replay.make_reverb_prioritized_sequence_replay(
#     env_spec, batch_size=batch_size)
# reverb_replay = make_reverb_episode_replay(
#     env_spec, max_sequence_length=max_game_length
# )


# In[ ]:


reverb_replay.adder.signature(env_spec)


# In[ ]:


actor = mz.actor.RandomActor(reverb_replay.adder)
learner = mz.learner.MooZiLearner(
    network=network,
    loss_fn=mz.loss.initial_inference_value_loss,
    optimizer=optimizer,
    data_iterator=reverb_replay.data_iterator,
    random_key=jax.random.PRNGKey(996),
)


# In[ ]:


agent = acme_agent.Agent(
    actor=actor, learner=learner, min_observations=100, observations_per_step=1)


# In[ ]:


loop = OpenSpielEnvironmentLoop(environment=env, actors=[agent])


# In[ ]:


num_episodes = 1000
result = loop.run(num_episodes=num_episodes)
print(result)


# In[ ]:


# replay signature
sig = reverb_replay.adder.signature(env_spec)
def recursive_as_dict(x):
    if hasattr(x, '_asdict'):
        x = x._asdict()
        return {k: recursive_as_dict(v) for k, v in x.items()}
    else:
        return x
pprint(recursive_as_dict(sig))


# In[ ]:


# from sacred import Experiment
# from sacred.observers import MongoObserver

# ex = Experiment('jupyter_ex', interactive=True)
# ex.observers.append(MongoObserver("172.17.0.6:27017"))


# In[ ]:


# @ex.config
# def my_config():
#     recipient = "world"
#     message = "Hello %s!" % recipient

# @ex.main
# def my_main(message):
#     print(message)


# In[ ]:


# acme.utils.loggers.


# In[ ]:






































