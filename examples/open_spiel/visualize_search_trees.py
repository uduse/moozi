# %%
from dataclasses import dataclass
from tqdm import tqdm
from PIL import Image
import chex
from flax import struct
from loguru import logger
import numpy as np
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import jax
from moozi.core.utils import fetch_device_array
from moozi.core.vis import BreakthroughVisualizer, visualize_search_tree
from moozi.gii import GII
from moozi.parameter_optimizer import load_params_and_state
from moozi.planner import Planner
from moozi.driver import Driver, get_config
import guild.ipy as guild

# %%
logger.info("Loading config")
config = get_config()

# %%
driver = Driver.setup(config)
vis = BreakthroughVisualizer(num_rows=config.env.num_rows, num_cols=config.env.num_cols)

planner = Planner(batch_size=1, model=driver.model, **config.vis.planner)
logger.info(f"Using planner {planner}")
logger.info("Loading checkpoints")
if config.vis.load.pkl:
    params, state = load_params_and_state(config.vis.load.pkl)
elif config.vis.load.run:
    df = guild.runs()
    df = df[
        df.run.map(lambda x: x.run["id"]).str.contains(config.vis.load.run)
    ].reset_index(drop=True)
    if len(df) == 1:
        latest_pkl = Path(df.run[0].run.dir) / "checkpoints/latest.pkl"
        params, state = load_params_and_state(latest_pkl)
    else:
        raise ValueError(f"Matches {len(df)} runs")
else:
    raise ValueError("load")

# %%
gii = GII(
    config.env.name,
    stacker=driver.stacker,
    planner=planner,
    params=params,
    state=state,
    random_key=jax.random.PRNGKey(0),
)


# @dataclass
# class ProjectionEntry:
#     hidden_state: chex.Array
#     step_index: int
#     game_index: int
#     last_action: int
#     next_action: int
#     image: Image.Image


entries = []
step_index = 0
game_index = 0
for i in tqdm(range(config.vis.num_steps), desc="making visualizations"):
    last_action = int(gii.env_feed.action)
    gii.tick()
    next_action = int(gii.env_feed.action)
    search_tree = gii.planner_out.tree
    hidden_state = gii.planner_out.tree.embeddings[0][0, 0].ravel()
    frame = fetch_device_array(gii.env_out.frame[0])
    search_tree = fetch_device_array(search_tree)
    root_state = gii.env.envs[0].backend.get_state
    vis_dir = Path(f"search_tree/{i}").resolve()
    if config.vis.show.tree:
        visualize_search_tree(vis, search_tree, root_state, vis_dir)
    # if config.vis.show.projection:
    #     entry = ProjectionEntry(
    #         hidden_state=hidden_state,
    #         # frame=frame,
    #         step_index=step_index,
    #         game_index=game_index,
    #         last_action=last_action,
    #         next_action=next_action,
    #         image=vis.make_image(frame),
    #     )
    #     if bool(gii.env_out.is_last):
    #         step_index = 0
    #         game_index += 1
    #     else:
    #         step_index += 1
    #     entries.append(entry)

if config.vis.show.projection:
    entries = fetch_device_array(entries)

    with open("embed.tsv", "w") as f:
        for e in entries:
            f.write("\t".join(map(str, e.hidden_state.tolist())))
            f.write("\n")

    with open("label.tsv", "w") as f:
        attrs = ["step_index", "game_index", "last_action", "next_action"]
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
        img = img.resize((image_height, image_width), Image.ANTIALIAS)
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
