#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import copy
from collections import namedtuple
from dataclasses import dataclass
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

import pyspiel
import open_spiel
from open_spiel.python import rl_environment
import dm_env
import acme
from acme.wrappers.open_spiel_wrapper import OpenSpielWrapper

from tqdm.notebook import tqdm
import numpy as np
import trueskill

import moozi as mz

import guild.ipy as guild


# In[2]:


jax.config.update('jax_platform_name', 'cpu')


# In[20]:


x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]


# In[21]:


np.mean(x)


# In[22]:


np.median(x)

