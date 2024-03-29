import moozi as mz
import numpy as np
import random
from moozi.core.env import GIIEnv
from moozi.core.link import link
from moozi.core.tape import make_tape
from moozi.core.trajectory_collector import TrajectoryCollector
from moozi.laws import make_vec_env
import pytest


ENV_NAMES = [
    "OpenSpiel:catch(rows=2,columns=3)",
    "OpenSpiel:go(board_size=9)",
    "OpenSpiel:breakthrough(rows=4,columns=4)",
    "MinAtar:Breakout-v1",
    "MinAtar:SpaceInvaders-v1",
]


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_gii_env(env_name):
    env = GIIEnv.new(env_name)
    assert env.spec
    env_feed = env.init()
    
    for i in range(100):
        env_out = env.step(env_feed)
        assert env_out
        action = np.random.choice(np.flatnonzero(env_out.legal_actions))
        env_feed = env_feed.replace(action=action, reset=env_out.is_last)


# @pytest.mark.parametrize("env_name", ENV_NAMES)
# def test_vec_env(env_name):
#     vec_env = make_vec_env(env_name, num_envs=2)
#     tape = make_tape()
#     tape.update(vec_env.malloc())
#     link(vec_env.apply)(tape)


# def test_gii_breakthrough():
#     env = GIIEnv("OpenSpiel:breakthrough(rows=5,columns=6)")
#     num_rows, num_cols = env._backend.observation_spec().observation.shape[:2]
#     vis = BreakthroughVisualizer(num_rows, num_cols)

#     feed = env.init()
#     imgs = []
#     for i in range(100):
#         env_out = env.step(feed)
#         feed.action = np.random.choice(np.argwhere(env_out.legal_actions).ravel())
#         feed.reset = env_out.is_last
#         img = vis.make_image(env_out.frame)
#         imgs.append(img)
#     save_gif(imgs)
