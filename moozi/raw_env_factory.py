from acme.wrappers import OpenSpielWrapper, SinglePrecisionWrapper
from acme.specs import make_environment_spec
import open_spiel
from absl import logging


def make_catch():
    prev_verbosity = logging.get_verbosity()
    logging.set_verbosity(logging.WARNING)

    env_columns, env_rows = 6, 6
    raw_env = open_spiel.python.rl_environment.Environment(
        f"catch(columns={env_columns},rows={env_rows})"
    )
    env = OpenSpielWrapper(raw_env)
    env = SinglePrecisionWrapper(env)
    env_spec = make_environment_spec(env)

    logging.set_verbosity(prev_verbosity)
    return env, env_spec


def make_tic_tac_toe():
    raw_env = open_spiel.python.rl_environment.Environment(f"tic_tac_toe")
    env = OpenSpielWrapper(raw_env)
    env = SinglePrecisionWrapper(env)
    env_spec = make_environment_spec(env)
    return env, env_spec


# make_env = make_catch


def make_openspiel_env(str):
    prev_verbosity = logging.get_verbosity()
    logging.set_verbosity(logging.WARNING)

    env_columns, env_rows = 6, 6
    raw_env = open_spiel.python.rl_environment.Environment(str)
    env = OpenSpielWrapper(raw_env)
    env = SinglePrecisionWrapper(env)
    env_spec = make_environment_spec(env)

    logging.set_verbosity(prev_verbosity)
    return env, env_spec


def make_env(str):
    try:
        env = make_openspiel_env(str)
        return env
    except:
        pass
