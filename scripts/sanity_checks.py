#!/usr/bin/env python
# coding: utf-8

# %%
get_ipython().system("find /usr/ | grep libcuda")
get_ipython().system("nvidia-smi")
get_ipython().system("nvcc --version")


# %%

import jax

print("jax\t", jax.devices())
