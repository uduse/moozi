# %% 
import dm_env
from dataclasses import dataclass
from moozi.core import make_env
import numpy as np
import pyspiel
import random

# %% 
env = make_env("OpenSpiel:breakthrough")


# %%
env.reset()

# %%
@dataclass
class MooZiEnv:
    name: str
    
    def init(self):
        return make_env()
    
    def apply(env: dm_env.Environment, is_last: bool, action: int):
        if is_last:
            timestep = env.reset()
        else:
            # action 0 is reserved for termination
            timestep = env.step(action - 1)

        if timestep.reward is None:
            reward = 0.0
        else:
            reward = timestep.reward

        legal_actions_mask = np.ones(env.action_space.n, dtype=np.float32)

        return dict(
            frame=np.asarray(timestep.observation, dtype=float),
            is_first=np.asarray(timestep.first(), dtype=bool),
            is_last=np.asarray(timestep.last(), dtype=bool),
            to_play=np.asarray(0, dtype=int),
            reward=np.asarray(reward, dtype=float),
            legal_actions_mask=np.asarray(legal_actions_mask, dtype=int),
        )

    return Law(
        name="dm_env",
        malloc=malloc,
        # this is not linked yet
        apply=apply,
        read=get_keys(apply),
    )
