#!/usr/bin/env python
# coding: utf-8

# In[6]:


get_ipython().system('find /usr/ | grep libcuda')
get_ipython().system('nvidia-smi')
get_ipython().system('nvcc --version')


# In[7]:


import jax
print('jax\t', jax.devices())


# In[8]:


import torch
print('torch\t', end='')
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
else:
    print('Not Found.')
print('\n\n')


# In[9]:


import pymongo

MONGODB_IP = "172.17.0.6"
client = pymongo.MongoClient(MONGODB_IP, 27017)
if client.db.command('ping')['ok'] == 1:
    print("Mongo DB -> OK")
else:
    print("Mongo DB -> FAILED")

