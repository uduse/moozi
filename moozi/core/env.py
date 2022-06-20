from dataclasses import dataclass
import numpy as np
import jax.numpy as jnp
import functools
from typing import List
from acme.utils.tree_utils import stack_sequence_fields
from loguru import logger
import uuid
import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import tree
import dm_env
from acme.wrappers import (
    OpenSpielWrapper,
    SinglePrecisionWrapper,
    EnvironmentWrapper,
    AtariWrapper,
    GymAtariAdapter,
    GymWrapper,
    wrap_all,
)
from acme.specs import make_environment_spec
import open_spiel
from absl import logging

from moozi.laws import MinAtarEnvLaw


class TransformObservationWrapper(EnvironmentWrapper):
    """Wrapper which converts environments from double- to single-precision."""

    def __init__(self, environment: dm_env.Environment, transform_fn):
        super().__init__(environment)
        self._transform_fn = transform_fn

    def _convert_timestep(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
        if isinstance(timestep.observation, list):
            obses = [
                self._transform_fn(olt.observation) for olt in timestep.observation
            ]
            new_olts = [
                olt._replace(observation=obses[i])
                for i, olt in enumerate(timestep.observation)
            ]
            return timestep._replace(observation=new_olts)
        else:
            raise NotImplementedError

    def step(self, action) -> dm_env.TimeStep:
        return self._convert_timestep(self._environment.step(action))

    def reset(self) -> dm_env.TimeStep:
        return self._convert_timestep(self._environment.reset())

    def observation_spec(self):
        olt = self._environment.observation_spec()
        dummy_obs = olt.observation.generate_value()
        transformed_shape = self._transform_fn(dummy_obs).shape
        updated_olt = olt._replace(
            observation=olt.observation.replace(shape=transformed_shape)
        )
        return updated_olt


def make_catch(num_rows=5, num_cols=5):
    env_name = f"catch(rows={num_rows},columns={num_cols})"
    return make_openspiel_env_and_spec(env_name)


def make_openspiel_env_and_spec(env_name):
    prev_verbosity = logging.get_verbosity()
    logging.set_verbosity(logging.WARNING)

    raw_env = open_spiel.python.rl_environment.Environment(env_name)

    if raw_env.name == "catch":
        game_params = raw_env.game.get_parameters()
        num_rows, num_cols = game_params["rows"], game_params["columns"]

        def transform_obs(obs):
            return obs.reshape((num_rows, num_cols, 1))

    elif raw_env.name == "tic_tac_toe":

        def transform_obs(obs):
            return obs.reshape((3, 3, 3)).swapaxes(0, 2)

    elif raw_env.name == "go":

        board_size = raw_env.game.get_parameters()["board_size"]

        def transform_obs(obs):
            return obs.reshape((board_size, board_size, 4))

    else:
        raise ValueError(f"Unknown OpenSpiel environment: {raw_env.name}")

    env = OpenSpielWrapper(raw_env)
    env = SinglePrecisionWrapper(env)
    env = TransformObservationWrapper(env, transform_obs)
    env_spec = make_environment_spec(env)

    logging.set_verbosity(prev_verbosity)

    return env, env_spec


def make_atari_env(level) -> dm_env.Environment:
    """Loads the Atari environment."""
    env = gym.make(level, full_action_space=True)

    # Always use episodes of 108k steps as this is standard, matching the paper.
    max_episode_len = 108000
    wrapper_list = [
        GymAtariAdapter,
        functools.partial(
            AtariWrapper,
            max_episode_len=max_episode_len,
            num_stacked_frames=1,
            to_float=True,
        ),
        SinglePrecisionWrapper,
    ]

    return wrap_all(env, wrapper_list)


def make_minatar_env(level) -> dm_env.Environment:
    env = gym.make("MinAtar/" + level)

    wrapper_list = [
        GymWrapper,
        SinglePrecisionWrapper,
    ]

    return wrap_all(env, wrapper_list)


def make_env_and_spec(env_name):
    if ":" in env_name:
        lib_type, env_name = env_name.split(":")
        if lib_type == "OpenSpiel":
            return make_openspiel_env_and_spec(env_name)

        elif lib_type == "Gym":
            import gym

            env = make_atari_env(env_name)
            return env, make_environment_spec(env)
        elif lib_type == "MinAtar":
            import gym

            env = make_minatar_env(env_name)
            return env, make_environment_spec(env)
        else:
            raise ValueError(f"Unknown library type: {lib_type}")
    else:
        try:
            ids = [x.id for x in gym.envs.registry.all()]
            if env_name in ids:
                env = make_atari_env(env_name)
                return env, make_environment_spec(env)
        except:
            pass

        try:
            return make_openspiel_env_and_spec(env_name)
        except:
            raise ValueError(f"Environment {env_name} not found")


def make_env(env_name):
    return make_env_and_spec(env_name)[0]


# environment specs should be the same if the `env_name` is the same
@functools.lru_cache(maxsize=None)
def make_spec(env_name):
    return make_env_and_spec(env_name)[1]


@dataclass
class VecEnv:
    env_name: str
    num_envs: int

    def malloc(self):
        envs = [make_env(self.env_name) for _ in range(self.num_envs)]
        dim_actions = envs[0].action_spec().num_values
        obs_shape = envs[0].observation_spec().shape
        envs = [MinAtarEnvLaw(env) for env in envs]
        action = np.full(self.num_envs, fill_value=0, dtype=jnp.int32)
        action_probs = jnp.full(
            (self.num_envs, dim_actions), fill_value=0, dtype=jnp.float32
        )
        obs = jnp.zeros((self.num_envs, *obs_shape), dtype=jnp.float32)
        is_first = jnp.full(self.num_envs, fill_value=False, dtype=bool)
        is_last = jnp.full(self.num_envs, fill_value=True, dtype=bool)
        to_play = jnp.zeros(self.num_envs, dtype=jnp.int32)
        reward = jnp.zeros(self.num_envs, dtype=jnp.float32)
        legal_actions_mask = jnp.ones((self.num_envs, dim_actions), dtype=jnp.int32)
        return {
            "envs": envs,
            "obs": obs,
            "action": action,
            "action_probs": action_probs,
            "is_first": is_first,
            "is_last": is_last,
            "to_play": to_play,
            "reward": reward,
            "legal_actions_mask": legal_actions_mask,
        }

    def __call__(self, envs, is_last: List[bool], action: List[int]):
        updates_list = []
        for env, is_last_, action_ in zip(envs, is_last, action):
            updates = env(is_last=is_last_, action=action_)
            updates_list.append(updates)
        return stack_sequence_fields(updates_list)
