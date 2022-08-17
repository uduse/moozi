import moozi as mz
from moozi.core.link import link
from moozi.core.tape import make_tape
from moozi.laws import make_vec_env
import pytest


ENV_NAMES = [
    "OpenSpiel:catch(rows=2,columns=3)",
    # "Gym:Pong-v0",
    "MinAtar:Breakout-v1",
    "MinAtar:SpaceInvaders-v1",
]


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_env(env_name):
    env, env_spec = mz.make_env_and_spec(env_name)
    env.reset()
    if env_name.split(":")[0] == "OpenSpiel":
        env.step([0])
    else:
        env.step(0)


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_vec_env(env_name):
    vec_env = make_vec_env(env_name, num_envs=2)
    tape = make_tape()
    tape.update(vec_env.malloc())
    link(vec_env.apply)(tape)

def test_gii_breakthrough():
    env = GIIEnv("OpenSpiel:breakthrough(rows=5,columns=6)")
    num_rows, num_cols = env._backend.observation_spec().observation.shape[:2]
    vis = BreakthroughVisualizer(num_rows, num_cols)

    feed = env.init()
    imgs = []
    for i in range(100):
        env_out = env.step(feed)
        feed.action = np.random.choice(np.argwhere(env_out.legal_actions).ravel())
        feed.reset = env_out.is_last
        img = vis.make_image(env_out.frame)
        imgs.append(img)
    save_gif(imgs)