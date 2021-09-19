# %%
from moozi.utils import SimpleQueue
from time import time
from typing import Callable, Dict, List, NamedTuple
from acme.agents.jax.impala.types import Observation
import numpy as np
import acme
import dm_env
import moozi as mz
import open_spiel
import ray
from acme.wrappers.open_spiel_wrapper import OpenSpielWrapper
from acme.wrappers import SinglePrecisionWrapper
import tree

# %%
# ray.init(ignore_reinit_error=True)

# %%


# %%
def _pad_frames(self, frames: List[np.ndarray]):
    orig_obs_shape = self._env.observation_spec().observation.shape
    while len(frames) < self._num_stacked_frames:
        padding = np.zeros(orig_obs_shape)
        frames.append(padding)
    return np.array(frames)


def env_factory():
    env_columns, env_rows = 2, 2
    raw_env = open_spiel.python.rl_environment.Environment(
        f"catch(columns={env_columns},rows={env_rows})"
    )
    env = OpenSpielWrapper(raw_env)
    env = SinglePrecisionWrapper(env)
    env_spec = acme.specs.make_environment_spec(env)
    return env, env_spec


# %%
env, env_spec = env_factory()

# %%
timestep = env.reset()

# %%
s = env.observation_spec()

# %%
EnvFactory = Callable[[], dm_env.Environment]


@ray.remote
class InteractionWorker(object):
    def __init__(self, env_factory: EnvFactory, player_shells_factory):
        self._env = env_factory()
        self._player_shells = player_shells_factory()


class ObservationStackingLayer(object):
    def __init__(self, num_stacked_frames: int):
        self._num_stacked_frames = num_stacked_frames
        self._stacked_frames = mz.utils.SimpleQueue(num_stacked_frames)


class PlayerShell(object):
    def __init__(self):
        pass


# %%
# InteractionWorker.remote(env_factory)

# %%
class Law(object):
    def read(self):
        pass

    def write(self):
        pass


class EnvironmentLaw(Law):
    def __init__(self) -> None:
        super().__init__()


class UniverseLoop(object):
    def __init__(self, materia: dict, laws: List[Law]):
        self._materia = materia
        self._laws = laws


# # %%


class EnvironmentMateria(NamedTuple):
    state: dm_env.Environment
    to_play: int
    is_first: bool
    is_last: bool


class PlayerMateria(NamedTuple):
    last_frames: SimpleQueue


class Materia(NamedTuple):
    env: EnvironmentMateria


# materia = {
#     "env": {
#         "state": None,
#         "to_play"
#     },
#     "players": [{"last_frames": None}],
# }


def environment_law(read):
    return {}


laws = []
UniverseLoop(materia, laws)
