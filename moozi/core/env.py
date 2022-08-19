from dataclasses import dataclass, field
from flax import struct
import pyspiel
import chex
import numpy as np
import jax.numpy as jnp
import functools
from typing import List
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
    StepLimitWrapper,
    wrap_all,
)
from acme.specs import make_environment_spec, EnvironmentSpec
import open_spiel
from absl import logging
from moozi.core import BASE_PLAYER
from moozi.core.utils import (
    stack_sequence_fields_pytree,
    unstack_sequence_fields_pytree,
)


class TransformFrameWrapper(EnvironmentWrapper):
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

        def transform_frame(frame):
            return frame.reshape((num_rows, num_cols, 1))

    elif raw_env.name == "tic_tac_toe":

        def transform_frame(frame):
            return frame.reshape((3, 3, 3)).swapaxes(0, 2)

    elif raw_env.name == "go":

        board_size = raw_env.game.get_parameters()["board_size"]

        def transform_frame(frame):
            return frame.reshape((board_size, board_size, 4))

    elif raw_env.name == "breakthrough":

        target_shape = raw_env.game.observation_tensor_shape()

        def transform_frame(frame):
            return np.moveaxis(frame.reshape(target_shape), 0, -1)

    else:
        raise ValueError(f"Game not support by MooZi: {raw_env.name}")

    env = OpenSpielWrapper(raw_env)
    env = SinglePrecisionWrapper(env)
    env = TransformFrameWrapper(env, transform_frame)
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


def make_minatar_env(level, **kwargs) -> dm_env.Environment:
    env = gym.make("MinAtar/" + level, sticky_action_prob=0.0)

    max_episode_len = 7200
    wrapper_list = [
        GymWrapper,
        functools.partial(
            StepLimitWrapper,
            step_limit=max_episode_len,
        ),
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


class GIIEnvFeed(struct.PyTreeNode):
    action: np.ndarray
    reset: np.ndarray


class GIIEnvOut(struct.PyTreeNode):
    frame: np.ndarray
    is_first: np.ndarray
    is_last: np.ndarray
    to_play: np.ndarray
    reward: np.ndarray
    legal_actions: np.ndarray

    def cast(self):
        return GIIEnvOut(
            frame=np.asarray(self.frame, dtype=np.float32),
            is_first=np.asarray(self.is_first, dtype=np.bool8),
            is_last=np.asarray(self.is_last, dtype=np.bool8),
            to_play=np.asarray(self.to_play, dtype=np.int32),
            reward=np.asarray(self.reward, dtype=np.float32),
            legal_actions=np.asarray(self.legal_actions, dtype=np.int32),
        )


class GIIEnv(struct.PyTreeNode):
    name: str
    num_players: int
    spec: EnvironmentSpec
    backend: dm_env.Environment = struct.field(pytree_node=False)

    @staticmethod
    def new(env_name):
        """Smart constructor."""
        backend, spec = make_env_and_spec(env_name)
        num_players = backend.num_players
        return GIIEnv(
            name=env_name,
            num_players=num_players,
            spec=spec,
            backend=backend,
        )

    @property
    def dim_action(self):
        return self.spec.actions.num_values + 1

    def init(self) -> GIIEnvFeed:
        return GIIEnvFeed(action=0, reset=True)

    def step(self, input_: GIIEnvFeed) -> GIIEnvOut:
        if input_.reset:
            timestep = self.backend.reset()
        else:
            # action 0 is reserved for termination
            timestep = self.backend.step([input_.action - 1])

        to_play = self.backend.current_player
        frame = self._get_frame(timestep, to_play)
        reward = self._get_reward(timestep, to_play)
        legal_actions = self._get_legal_actions(timestep, to_play)
        return GIIEnvOut(
            frame=frame,
            is_first=timestep.first(),
            is_last=timestep.last(),
            to_play=to_play,
            reward=reward,
            legal_actions=legal_actions,
        ).cast()

    @staticmethod
    def _get_reward(timestep: dm_env.TimeStep, to_play: int):
        if timestep.reward is None:
            return 0.0
        elif isinstance(timestep.reward, np.ndarray):
            return timestep.reward[BASE_PLAYER]
        else:
            raise ValueError

    @staticmethod
    def _get_frame(timestep: dm_env.TimeStep, to_play: int):
        if to_play == pyspiel.PlayerId.TERMINAL:
            return timestep.observation[0].observation
        else:
            return timestep.observation[to_play].observation

    @staticmethod
    def _get_legal_actions(timestep: dm_env.TimeStep, to_play: int):
        assert isinstance(timestep.observation, list)
        if to_play == pyspiel.PlayerId.TERMINAL:
            la = np.zeros_like(timestep.observation[0].legal_actions)
            return np.insert(la, 0, [1])
        else:
            la = timestep.observation[to_play].legal_actions
            return np.insert(la, 0, [0])


# TODO: extra dummy action handling into a class


class GIIVecEnv(struct.PyTreeNode):
    name: str
    num_envs: int
    num_players: int
    envs: List[GIIEnv] = struct.field(pytree_node=False)

    @staticmethod
    def new(env_name: str, num_envs: int):
        assert num_envs >= 1
        envs = [GIIEnv.new(env_name) for _ in range(num_envs)]
        num_players = envs[0].num_players
        return GIIVecEnv(
            name=env_name,
            num_envs=num_envs,
            num_players=num_players,
            envs=envs,
        )

    def init(self) -> GIIEnvFeed:
        return stack_sequence_fields_pytree([env.init() for env in self.envs])

    def step(self, input_: GIIEnvFeed) -> GIIEnvOut:
        input_list = unstack_sequence_fields_pytree(input_, self.num_envs)
        output_list = tree.map_structure_with_path(
            lambda path, env: env.step(input_list[path[0]]), self.envs
        )
        return stack_sequence_fields_pytree(output_list)
