# %%
from PIL import Image
import seaborn as sns
from IPython.display import display
import cv2
import random
from dataclasses import dataclass
import numpy as np
import moozi as mz
import dm_env


@dataclass
class MinAtarEnvLaw:
    env: dm_env.Environment

    def __call__(self, is_last, action: int):
        if is_last:
            timestep = self.env.reset()
        else:
            timestep = self.env.step(action)

        if timestep.reward is None:
            reward = 0.0
        else:
            reward = timestep.reward

        legal_actions_mask = np.ones(self.env.action_space.n, dtype=np.float32)

        return dict(
            obs=np.array(timestep.observation, dtype=float),
            is_first=timestep.first(),
            is_last=timestep.last(),
            to_play=0,
            reward=reward,
            legal_actions_mask=legal_actions_mask,
        )


def make_min_atar_gif_recorder(n_channels=6):
    cmap = sns.color_palette("cubehelix", n_channels)
    cmap.insert(0, (0, 0, 0))
    cs = np.array([cmap[i] for i in range(n_channels + 1)])

    def malloc():
        return {"images": []}

    def forward(is_last, obs, images):
        numerical_state = np.array(
            np.amax(obs * np.reshape(np.arange(n_channels) + 1, (1, 1, -1)), 2) + 0.5,
            dtype=int,
        )
        rgbs = np.array(cs[numerical_state - 1] * 255, dtype=np.uint8)
        img = Image.fromarray(rgbs)
        img = img.resize((img.width * 10, img.height * 10), Image.BOX)
        images = images + [img]
        if is_last and images:
            images[0].save(
                "ani.gif",
                save_all=True,
                append_images=images[1:],
                optimize=False,
                duration=40,
                # loop=0,
            )
            images = []
        return {"images": images}

    return malloc, forward


# %%
env = mz.make_env("MinAtar:SpaceInvaders-v1")
env_law = MinAtarEnvLaw(env, record_video=True)
is_last = True

env = mz.make_env("MinAtar:SpaceInvaders-v1")
env_law = MinAtarEnvLaw(env, record_video=True)
is_last = True
images = []
for _ in range(100):
    action = random.choice(list(range(4)))
    # Image.new("RGB", size=(200, 200))
    obs, is_fisrt, is_last, to_play, reward, legal = env_law(is_last, action).values()
    numerical_state = np.array(
        np.amax(obs * np.reshape(np.arange(n_channels) + 1, (1, 1, -1)), 2) + 0.5,
        dtype=int,
    )
    rgbs = np.array(cs[numerical_state - 1] * 255, dtype=np.uint8)
    img = Image.fromarray(rgbs)
    img = img.resize((img.width * 10, img.height * 10), Image.BOX)
    images.append(img)

images[0].save(
    "ani.gif",
    save_all=True,
    append_images=images[1:],
    optimize=False,
    duration=40,
    # loop=0,
)
# %%

width = 200
center = width // 2
color_1 = (0, 0, 0)
color_2 = (255, 255, 255)
max_radius = int(center * 1.5)
step = 8

images[0].save(
    "ani.gif",
    save_all=True,
    append_images=images[1:],
    optimize=False,
    duration=40,
    loop=0,
)
