import functools
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
    wrap_all,
)
from acme.specs import make_environment_spec
import open_spiel
from absl import logging


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
    prev_verbosity = logging.get_verbosity()
    logging.set_verbosity(logging.WARNING)

    def transform_obs(obs):
        return obs.reshape((num_rows, num_cols, 1))

    raw_env = open_spiel.python.rl_environment.Environment(
        f"catch(rows={num_rows},columns={num_cols})"
    )

    env = OpenSpielWrapper(raw_env)
    env = SinglePrecisionWrapper(env)
    env = TransformObservationWrapper(env, transform_obs)
    env_spec = make_environment_spec(env)

    logging.set_verbosity(prev_verbosity)
    return env, env_spec


def make_tic_tac_toe():
    prev_verbosity = logging.get_verbosity()
    logging.set_verbosity(logging.WARNING)

    def transform_obs(obs):
        return obs.reshape((3, 3, 3)).swapaxes(0, 2)

    raw_env = open_spiel.python.rl_environment.Environment(f"tic_tac_toe")
    env = OpenSpielWrapper(raw_env)
    env = SinglePrecisionWrapper(env)
    env = TransformObservationWrapper(env, transform_obs)
    env_spec = make_environment_spec(env)

    logging.set_verbosity(prev_verbosity)
    return env, env_spec


def make_openspiel_env_and_spec(str):
    prev_verbosity = logging.get_verbosity()
    logging.set_verbosity(logging.WARNING)
    raw_env = open_spiel.python.rl_environment.Environment(str)
    env = OpenSpielWrapper(raw_env)
    env = SinglePrecisionWrapper(env)
    env_spec = make_environment_spec(env)

    logging.set_verbosity(prev_verbosity)
    return env, env_spec


def make_atari_env(
    level: str = "PongNoFrameskip-v4",
) -> dm_env.Environment:
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
    ]
    wrapper_list.append(SinglePrecisionWrapper)

    return wrap_all(env, wrapper_list)


def make_env_and_spec(str):
    try:
        import gym
        ids = [x.id for x in gym.envs.registry.all()]
        if str in ids:
            env = make_atari_env(str)
            return env, make_environment_spec(env)
    except:
        pass

    try:
        env, env_spec = make_openspiel_env_and_spec(str)
        if env.name == "catch":
            game_params = env.environment._environment._game.get_parameters()
            num_rows, num_cols = game_params["rows"], game_params["columns"]
            return make_catch(num_rows=num_rows, num_cols=num_cols)
        elif env.name == "tic_tac_toe":
            return make_tic_tac_toe()
        else:
            return env, env_spec
    except:
        raise ValueError(f"Environment {str} not found")


def make_env(str):
    return make_env_and_spec(str)[0]


# environment specs should be the same if the `str` is the same
@functools.lru_cache(maxsize=None)
def make_env_spec(str):
    return make_env_and_spec(str)[1]
