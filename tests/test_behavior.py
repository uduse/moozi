from moozi import make_env
from moozi.core.env import make_env_and_spec


def test_obs():
    env_name = "MinAtar:SpaceInvaders-v1"
    _, env_spec = make_env_and_spec(env_name)
