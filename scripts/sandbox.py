# %%
import collections
import copy
import dataclasses
import enum
import functools
import inspect
import types
from time import time
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Union

import acme
import dm_env
import jax
import moozi as mz
import numpy as np
import open_spiel
import ray
import tree
from acme.agents.jax.impala.types import Observation
from acme.jax.networks.base import Value
from acme.wrappers import SinglePrecisionWrapper
from acme.wrappers.open_spiel_wrapper import OpenSpielWrapper
from moozi.utils import SimpleQueue

# %%
ray.init(ignore_reinit_error=True)

# %%
# def env_factory():
#     env_columns, env_rows = 5, 5
#     raw_env = open_spiel.python.rl_environment.Environment(
#         f"catch(columns={env_columns},rows={env_rows})"
#     )
#     env = OpenSpielWrapper(raw_env)
#     env = SinglePrecisionWrapper(env)
#     env_spec = acme.specs.make_environment_spec(env)
#     return env, env_spec


def env_factory():
    raw_env = open_spiel.python.rl_environment.Environment(f"tic_tac_toe")
    env = OpenSpielWrapper(raw_env)
    env = SinglePrecisionWrapper(env)
    env_spec = acme.specs.make_environment_spec(env)
    return env, env_spec


# @ray.remote
# class InteractionWorker(object):
#     def __init__(self, env_factory: EnvFactory, player_shells_factory):
#         self._env = env_factory()
#         self._player_shells = player_shells_factory()


# class ObservationStackingLayer(object):
#     def __init__(self, num_stacked_frames: int):
#         self._num_stacked_frames = num_stacked_frames
#         self._stacked_frames = mz.utils.SimpleQueue(num_stacked_frames)


# class PlayerShell(object):
#     def __init__(self):
#         pass


# %%

# %%
@dataclasses.dataclass
class Artifact:
    # meta
    num_ticks: int = 0

    # environment
    env_state: dm_env.Environment = None
    timestep: dm_env.TimeStep = None
    to_play: int = -1
    action: int = -1

    # player
    last_frames: Optional[SimpleQueue] = None


class _Link(object):
    def __init__(
        self,
        func: Callable[..., Optional[dict]],
        to_read: Union[List[str], str] = "auto",
        to_write: Union[List[str], str] = "auto",
    ) -> None:
        self._func = func
        self._to_read = to_read
        self._to_write = to_write

        functools.update_wrapper(self, func)

    @staticmethod
    def _read_artifact(artifact, keys_to_read: List[str]):
        return {key: getattr(artifact, key) for key in keys_to_read}

    @staticmethod
    def _update_artifact(artifact, updates: Dict[str, Any]):
        for key, val in updates.items():
            setattr(artifact, key, val)

    @staticmethod
    def _artifact_has_keys(artifact, keys: List[str]) -> bool:
        return set(keys) <= set(artifact.__dict__.keys())

    def __call__(self, artifact: Artifact):
        keys_to_read = self._get_keys_to_read(artifact)
        artifact_window = _Link._read_artifact(artifact, keys_to_read)
        updates = self._func(**artifact_window)
        if not updates:
            updates = {}
        self._validate_updates(artifact, updates)
        _Link._update_artifact(artifact, updates)
        return updates

    def _validate_updates(self, artifact, updates):
        if not _Link._artifact_has_keys(artifact, list(updates.keys())):
            raise ValueError

        if self._to_write == "auto":
            pass
        elif isinstance(self._to_write, list):
            update_nothing = (not self._to_write) and (not updates)
            if update_nothing:
                pass
            elif self._to_write != list(updates.keys()):
                raise ValueError("write_view keys mismatch.")
        else:
            raise ValueError("`to_write` type not accepted.")

    def _get_keys_to_read(self, artifact):
        if self._to_read == "auto":
            keys = list(inspect.signature(self._func).parameters.keys())
            if not _Link._artifact_has_keys(artifact, keys):
                raise ValueError(f"{str(keys)} not in {str(artifact.__dict__.keys())})")
        elif isinstance(self._to_read, list):
            keys = self._to_read
        else:
            raise ValueError("`to_read` type not accepted.")
        return keys


def link(*args, **kwargs):
    if len(args) == 1 and not kwargs and callable(args[0]):
        func = args[0]
        return _Link(func, to_read="auto", to_write="auto")
    else:
        func = functools.partial(_Link, **kwargs)
        return func


@link
def print_timestep(timestep):
    print(timestep)


@link
def random_action_law(timestep: dm_env.TimeStep):
    if not timestep.last():
        legal_actions = timestep.observation[0].legal_actions
        random_action = np.random.choice(np.flatnonzero(legal_actions == 1))
        return {"action": random_action}
    else:
        return {"action": -1}


def tick(artifact, linked_laws):
    for law in linked_laws:
        law(artifact)


@link
def environment_law(
    env_state: dm_env.Environment, timestep: dm_env.TimeStep, action: int
):
    if timestep is None or timestep.last():
        timestep = env_state.reset()
    else:
        timestep = env_state.step([action])
    return {"env_state": env_state, "timestep": timestep}


@link
def increment_tick(num_ticks):
    return {"num_ticks": num_ticks + 1}


class UniverseWorker(object):
    def __init__(self, artifact_factory, laws_factory):
        self._artifact = artifact_factory()
        self._laws = laws_factory()

    def tick(self):
        tick(self._artifact, self._laws)

    def loop(self):
        import time

        while True:
            time.sleep(1)
            self.tick()

    def artifact(self):
        return self._artifact

    def laws(self):
        return self._laws

def make_player_shell(player: int):
    

# %%
def artifact_factory():
    artifact = Artifact()
    artifact.env_state = env_factory()[0]
    return artifact


def laws_factory():
    return [
        environment_law,
        random_action_law,
        link(lambda timestep: print("reward:", timestep.reward)),
        make_say_hello(0),
        make_say_hello(1),
        increment_tick,
    ]


# %%
worker = UniverseWorker(artifact_factory, laws_factory)
remote_worker_fn = lambda: ray.remote(UniverseWorker).remote(
    artifact_factory, laws_factory
)
remote_worker = remote_worker_fn()

# %%
ref = remote_worker.tick.remote()

# %%
ray.get(ref)
# ray.kill(remote_worker)
# %%
# ray.get(remote_worker.artifact.remote())

# %%

# %%
raw_env = open_spiel.python.rl_environment.Environment(f"tic_tac_toe")
env = acme.wrappers.open_spiel_wrapper.OpenSpielWrapper(raw_env)
env = acme.wrappers.SinglePrecisionWrapper(env)
env_spec = acme.specs.make_environment_spec(env)

# %%
num_stacked_frames = 1
dim_repr = 64
dim_action = env_spec.actions.num_values
frame_shape = env_spec.observations.observation.shape
stacked_frame_shape = (num_stacked_frames,) + frame_shape
nn_spec = mz.nn.NeuralNetworkSpec(
    stacked_frames_shape=stacked_frame_shape,
    dim_repr=dim_repr,
    dim_action=dim_action,
    repr_net_sizes=(128, 128),
    pred_net_sizes=(128, 128),
    dyna_net_sizes=(128, 128),
)
network = mz.nn.get_network(nn_spec)
# %%

master_key = jax.random.PRNGKey(0)
network.init(master_key)
# %%


