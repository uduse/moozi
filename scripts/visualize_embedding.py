# %%
from sklearn.manifold import TSNE
import matplotlib.pylab as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from dataclasses import dataclass, replace
from tqdm import tqdm
from PIL import Image, ImageOps
import chex
from flax import struct
from loguru import logger
import numpy as np
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import jax
from moozi.core.utils import fetch_device_array
from moozi.core.vis import BreakthroughVisualizer, Visualizer, visualize_search_tree
from moozi.gii import GII
from moozi.nn.nn import TransitionFeatures
from moozi.parameter_optimizer import load_params_and_state
from moozi.planner import Planner
from moozi.driver import ConfigFactory, Driver, get_config
import guild.ipy as guild

# %%
logger.info("Loading config")
config = get_config("~/moozi/examples/open_spiel/config.yml")
pkl_path = "/home/zeyi/miniconda3/envs/moozi/.guild/runs/e95902811f16497c80397281a535f341/checkpoints/284800.pkl"
factory = ConfigFactory(config)

# %%
env = factory.make_env()
vis = Visualizer.match_env(env)
planner = factory.make_testing_planner()
planner = planner.replace(num_simulations=8, kwargs={**planner.kwargs, **{"dirichlet_fraction": 0.5, "dirichlet_alpha": 1.0}})
logger.info(f"Using planner {planner}")

# %%
df = guild.runs()
df = df[
    df.run.map(lambda x: x.run["id"]).str.contains(config.vis.load.run)
].reset_index(drop=True)
if len(df) == 1:
    latest_pkl = Path(df.run[0].run.dir) / "checkpoints/latest.pkl"
    params, state = load_params_and_state(latest_pkl)
else:
    raise ValueError(f"Matches {len(df)} runs")

# %%
params, state = load_params_and_state(pkl_path)

# %%
@dataclass
class ProjectionEntry:
    id_: int
    hidden_state: chex.Array
    step_index: int
    game_index: int
    value: float
    last_action: int
    next_action: int
    to_play: int
    image: Image.Image
    hidden_index: int = 0
    triple: int = -1


gii = GII(
    env=env,
    stacker=factory.make_history_stacker(),
    planner=planner,
    params=params,
    state=state,
    random_key=jax.random.PRNGKey(0),
)

# %%
entries: List[ProjectionEntry] = []
step_index = 0
game_index = 0
for i in tqdm(range(3000)):
    last_action = int(gii.action.action)
    gii.tick()
    next_action = int(gii.action.action)
    search_tree = gii.planner_out.tree
    hidden_state = np.array(gii.planner_out.tree.embeddings[0][0, 0]).ravel()
    frame = fetch_device_array(gii.env_out.frame[0])
    search_tree = fetch_device_array(search_tree)
    root_state = gii.env.envs[0].backend.get_state
    vis_dir = Path(f"search_tree/{i}").resolve()
    entry = ProjectionEntry(
        id_=i,
        hidden_state=hidden_state,
        step_index=step_index,
        game_index=game_index,
        last_action=last_action,
        next_action=next_action,
        to_play=int(gii.env_out.to_play),
        value=float(gii.planner_out.root_value),
        image=vis.make_image(frame),
    )
    if bool(gii.env_out.is_last):
        step_index = 0
        game_index += 1
    else:
        step_index += 1
    entries.append(entry)

# %%
def trace_hidden():
    for i, e in tqdm(
        enumerate(entries[:50])
    ):
        if e.next_action != 0:
            x = e.hidden_state
            nn_out, _ = gii.planner.model.trans_inference_unbatched(
                params=gii.params,
                state=gii.state,
                feats=TransitionFeatures(hidden_state=x, action=e.next_action),
                is_training=False,
            )
            im = Image.new(e.image.mode, e.image.size, color="white")
            im = ImageOps.expand(im, border=6, fill="black")

            triple = e.id_
            entry = ProjectionEntry(
                id_=len(entries),
                hidden_state=nn_out.hidden_state,
                step_index=e.step_index,
                game_index=e.game_index,
                last_action=e.next_action,
                value=float(nn_out.value),
                to_play=np.logical_not(e.to_play),
                next_action=0,
                image=im,
                hidden_index=1,
                triple=triple,
            )
            e.triple = triple
            entries.append(replace(entries[i + 1], triple=i))
            entries.append(entry)

entries = fetch_device_array(entries)

# %% 
x = np.array([e.hidden_state.ravel() for e in entries])
tsne_2d = TSNE(
    n_components=2, learning_rate="auto", init="random", perplexity=15
).fit_transform(x)

# %%
player_map = {
    0: "Black",
    1: "White",
    -4: "Terminal"
}
df = pd.DataFrame(
    {
        "step_index": [e.step_index for e in entries],
        "last_action": [e.last_action for e in entries],
        "next_action": [e.next_action for e in entries],
        "value": [e.value for e in entries],
        "to_play": [player_map[e.to_play] for e in entries],
        "2d_x": tsne_2d[:, 0],
        "2d_y": tsne_2d[:, 1],
    }
)

# %% 
sns.set(rc={'figure.figsize':(8.5, 11), 'figure.dpi': 300}, font_scale=1.0)
sns.set_style('dark')
fig, axes = plt.subplots(nrows=3, sharex=True, )
fig.tight_layout()
color_gradient = sns.cubehelix_palette(start=0.5, rot=-0.5, as_cmap=True)
color_diverge = sns.diverging_palette(240, 10, as_cmap=True)
x = np.array([e.hidden_state.ravel() for e in entries])

plot = sns.scatterplot(
    x="2d_x",
    y="2d_y",
    hue="value",
    palette=color_diverge,
    data=df,
    ax=axes[0],
)

plot = sns.scatterplot(
    x="2d_x",
    y="2d_y",
    hue="step_index",
    palette=color_gradient,
    data=df,
    ax=axes[1]
)

plot = sns.scatterplot(
    x="2d_x",
    y="2d_y",
    hue="to_play",
    data=df,
    ax=axes[2]
)

titles = ['Value', 'Move Number', "Player"]
for title, ax in zip(titles, axes):
    ax.legend(title=title, bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
    ax.set(xlabel=None, ylabel=None)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

plt.subplots_adjust(hspace=0.05)

# %%
entries = fetch_device_array(entries)

with open("embed.tsv", "w") as f:
    for e in entries:
        f.write("\t".join(map(str, e.hidden_state.ravel().tolist())))
        f.write("\n")

with open("label.tsv", "w") as f:
    attrs = [
        "step_index",
        "game_index",
        "last_action",
        "next_action",
        "hidden_index",
        "triple",
        "value",
    ]
    f.write("\t".join(attrs) + "\n")
    for e in entries:
        f.write("\t".join([str(getattr(e, attr)) for attr in attrs]) + "\n")

import math

images = [e.image for e in entries]
grid = int(math.sqrt(len(images))) + 1
# tensorboard supports sprite images up to 8192 x 8192
image_height = int(8192 / grid)
image_width = int(8192 / grid)

sprite = Image.new(
    mode="RGB", size=(image_width * grid, image_height * grid), color=(0, 0, 0)
)

for i in range(len(images)):
    row = int(i / grid)
    col = int(i % grid)
    img = images[i]
    img = img.resize((image_height, image_width), Image.Resampling.LANCZOS)
    row_loc = row * image_height
    col_loc = col * image_width
    sprite.paste(img, (col_loc, row_loc))

sprite.save("sprite.jpg")

projector_config = f"""
embeddings {{
tensor_path: "embed.tsv"
metadata_path: "label.tsv"
sprite {{
    image_path: "sprite.jpg"
    single_image_dim: {image_height}
    single_image_dim: {image_width}
}}
}}
"""

with open("projector_config.pbtxt", "w") as f:
    f.write(projector_config)


# %%
