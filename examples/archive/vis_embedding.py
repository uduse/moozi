# %%
import tree
from acme.jax.utils import add_batch_dim
from acme.utils.tree_utils import stack_sequence_fields, unstack_sequence_fields
from omegaconf import OmegaConf
import numpy as np
from IPython.display import display
from loguru import logger
import ray
from functools import partial
from moozi.core.tape import make_tape
from moozi.laws import MinAtarVisualizer
from moozi.nn.training import make_target_from_traj, _make_obs_from_train_target

from moozi.replay import ReplayBuffer
from moozi.parameter_optimizer import ParameterServer
from moozi.rollout_worker import RolloutWorker
from moozi.laws import *
from moozi.planner import convert_tree_to_graph, make_planner
from moozi.nn import RootFeatures, TransitionFeatures

from lib import (
    training_suite_factory,
    make_test_worker_universe,
    make_reanalyze_universe,
    make_env_worker_universe,
    get_config,
)

# %%

jax.config.update("jax_disable_jit", False)
config = get_config(
    {
        "test_worker.planner.num_simulations": 50,
        "test_worker.planner.output_tree": True,
        "test_worker.planner.limit_depth": True,
    },
    path="/home/zeyi/miniconda3/envs/moozi/.guild/runs/8c066392ab9e4b4aa22598b2a93da050/.guild/sourcecode/examples/minatar_space_invaders/config.yml",
)
print(OmegaConf.to_yaml(config))

# %%
ps = ParameterServer(training_suite_factory=training_suite_factory(config))
rb = ReplayBuffer(**config.replay.kwargs)
vis = MinAtarVisualizer()

# %%
weights_path = "/home/zeyi/miniconda3/envs/moozi/.guild/runs/8c066392ab9e4b4aa22598b2a93da050/checkpoints/42840.pkl"
ps.restore(weights_path)

# %%
u = make_test_worker_universe(config)

# %%
u.tape["params"] = ps.get_params()
u.tape["state"] = ps.get_state()
counter = 0
traj_counter = 0

# %%
hiddens = []
labels = []
images = []
last_actions = []

# %%
u.tape["random_key"] = jax.random.PRNGKey(0)
u.tape["is_last"] = np.array([True])

# %%
for i in range(10000):
    counter += 1
    print(counter)
    u.tick()
    hidden = u.tape["tree"].embeddings[0, 0].ravel()
    hiddens.append(hidden)

    last_actions.append(int(u.tape['action'][0]))
    if u.tape["is_first"][0]:
        traj_counter += 1
    labels.append(traj_counter)

    image = vis.make_image(u.tape["frame"][0])
    images.append(image)

# %%
import math

grid = int(math.sqrt(len(images))) + 1
image_height = int(8192 / grid)  # tensorboard supports sprite images up to 8192 x 8192
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
    sprite.paste(
        img, (col_loc, row_loc)
    ) 
    print(row_loc, col_loc)

sprite.save("sprite.jpg")

# %%
print(f"{image_width=}")

# %%
with open("embed.tsv", "w") as f:
    for hidden in hiddens:
        f.write("\t".join(map(str, hidden.tolist())))
        f.write("\n")

# %%
with open("label.tsv", "w") as f:
    f.write("index\tepisode\tframe\taction\n")
    last_label = labels[0]
    counter = 0
    for i, (label, action) in enumerate(zip(labels, last_actions)):
        if last_label != label:
            counter = 0
        f.write(f"{i}\t{label}\t{counter}\t{action}")
        f.write("\n")
        counter += 1
        last_label = label

# %%
