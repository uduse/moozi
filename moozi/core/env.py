import functools
import tree
import dm_env
from acme.wrappers import OpenSpielWrapper, SinglePrecisionWrapper, EnvironmentWrapper
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


def make_catch():
    prev_verbosity = logging.get_verbosity()
    logging.set_verbosity(logging.WARNING)

    env_rows = 5
    env_columns = 7

    def transform_obs(obs):
        return obs.reshape((env_rows, env_columns, 1))

    raw_env = open_spiel.python.rl_environment.Environment(
        f"catch(rows={env_rows},columns={env_columns})"
    )

    env = OpenSpielWrapper(raw_env)
    env = SinglePrecisionWrapper(env)
    env = TransformObservationWrapper(env, transform_obs)
    env_spec = make_environment_spec(env)

    logging.set_verbosity(prev_verbosity)
    return env, env_spec


def make_tic_tac_toe():
    raw_env = open_spiel.python.rl_environment.Environment(f"tic_tac_toe")
    env = OpenSpielWrapper(raw_env)
    env = SinglePrecisionWrapper(env)
    env_spec = make_environment_spec(env)
    return env, env_spec


def make_openspiel_env(str):
    prev_verbosity = logging.get_verbosity()
    logging.set_verbosity(logging.WARNING)
    raw_env = open_spiel.python.rl_environment.Environment(str)
    env = OpenSpielWrapper(raw_env)
    env = SinglePrecisionWrapper(env)
    env_spec = make_environment_spec(env)

    logging.set_verbosity(prev_verbosity)
    return env, env_spec


def make_env_and_spec(str):
    try:
        env, env_spec = make_openspiel_env(str)
        return env, env_spec
    except:
        raise ValueError(f"Environment {str} not found")


def make_env(str):
    env, _ = make_openspiel_env(str)
    return env


@functools.lru_cache(maxsize=None)
def make_env_spec(str):
    _, env_spec = make_openspiel_env(str)
    return env_spec
