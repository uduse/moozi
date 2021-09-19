# %%
import copy
import dataclasses
import inspect
import functools
import enum
import types
import collections

from acme.jax.networks.base import Value
from moozi.utils import SimpleQueue
from time import time
from typing import Any, Callable, Dict, List, NamedTuple, Optional
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
from enum import Enum, auto


# class ArtifactKeys(Enum):
#     env_state = auto()

#     to_play = auto()
#     is_first = auto()
#     is_last = auto()
#     last_reward = auto()
#     action = auto()

#     last_frames = auto()


@dataclasses.dataclass
class Artifact:
    env_state: dm_env.Environment = None

    # env obs
    timestep: dm_env.TimeStep = None
    to_play: int = 0

    action: int = 0

    # player
    last_frames: Optional[SimpleQueue] = None

    @property
    def keys(self):
        return list(self.__dict__.keys())

    def read_view(self, keys: List[str]):
        return {key: getattr(self, key) for key in keys}

    def write_view(self, items: Dict[str, Any]):
        for key, val in items.items():
            setattr(self, key, val)


def law(*args, read="auto", write="auto"):
    def _wrapper(law):
        @functools.wraps(law)
        def _apply_law(artifact: "Artifact"):
            if read == "auto":
                keys = list(inspect.signature(law).parameters.keys())
                if not (set(keys) <= set(artifact.keys)):
                    raise ValueError(f"{str(keys)} not in {str(artifact.keys)})")
            elif isinstance(read, list):
                keys = read
            else:
                raise ValueError

            partial_artifact = artifact.read_view(keys)
            updates = law(**partial_artifact)

            if write == "auto":
                pass
            elif isinstance(write, list):
                if (not write) and (not updates):
                    pass
                elif write != list(updates.keys()):
                    raise ValueError("write_view keys mismatch.")
            else:
                raise ValueError

            if updates:
                artifact.write_view(updates)

            return updates

        return _apply_law

    if len(args) == 1 and callable(args[0]):
        return law()(args[0])
    else:
        return _wrapper


@law(write=["env_state", "timestep"])
def environment_law(
    env_state: dm_env.Environment, timestep: dm_env.TimeStep, action: int
):
    if timestep is None or timestep.last():
        timestep = env_state.reset()
    else:
        timestep = env_state.step([action])
    return {"env_state": env_state, "timestep": timestep}


@law
def timestep_viewer(timestep):
    print(timestep)


def universe_loop(artifact, laws, copy=False):
    if copy:
        artifact = copy.deepcopy(artifact)
    for law in laws:
        law(artifact)


# %%
env_state = env_factory()[0]
artifact = Artifact(env_state=env_state)

for _ in range(5):
    universe_loop(artifact, [environment_law, timestep_viewer])

# %%
