import moozi as mz
import pytest


@pytest.mark.parametrize(
    "env_str",
    [
        "OpenSpiel:catch(rows=2,columns=3)",
        # "Gym:Pong-v0",
        "MinAtar:Breakout-v1",
    ],
)
def test_env(env_str):
    env, env_spec = mz.make_env_and_spec(env_str)
    env.reset()
    if env_str.split(":")[0] == "OpenSpiel":
        env.step([0])
    else:
        env.step(0)
