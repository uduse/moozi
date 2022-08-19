from os import PathLike
from tqdm import tqdm
from pathlib import Path
from typing import List, Optional, Sequence, Union, Dict

import chex
import jax
import jax.numpy as jnp
import mctx
import numpy as np
import pygraphviz
import pyspiel
from loguru import logger
from moozi import BASE_PLAYER
from moozi.utils import get_project_root
from PIL import Image, ImageOps


class Visualizer:
    def __init__(self, num_rows: int, num_cols: int):
        self.num_rows = num_rows
        self.num_cols = num_cols

    def make_image(self, frame: np.ndarray) -> Image:
        raise NotImplementedError


class BreakthroughVisualizer(Visualizer):
    def __init__(self, num_rows: int, num_cols: int):
        super().__init__(num_rows=num_rows, num_cols=num_cols)
        root = get_project_root()
        self.white_pawn_img = Image.open(root / "assets/white_pawn.png").convert("RGBA")
        self.black_pawn_img = Image.open(root / "assets/black_pawn.png").convert("RGBA")
        self.white_tile_img = Image.open(root / "assets/white_tile.png").convert("RGBA")
        self.black_tile_img = Image.open(root / "assets/black_tile.png").convert("RGBA")
        assert (
            self.white_pawn_img.size
            == self.black_pawn_img.size
            == self.white_tile_img.size
            == self.black_tile_img.size
        )

        self.token_width = self.black_pawn_img.width
        self.token_height = self.black_pawn_img.height

    def make_image(self, frame: np.ndarray) -> Image:
        frame = np.asarray(frame, dtype=np.float32)
        assert frame.shape == (self.num_rows, self.num_cols, 3)

        img_width = self.token_width * self.num_cols
        img_height = self.token_height * self.num_rows
        img = Image.new("RGBA", (img_width, img_height), (255, 255, 255))

        for row_idx, row in enumerate(frame):
            for col_idx, col in enumerate(row):
                loc = (self.token_width * col_idx, self.token_height * row_idx)
                if (row_idx + col_idx) % 2 == 0:
                    img.paste(self.white_tile_img, loc, mask=self.white_tile_img)
                else:
                    img.paste(self.black_tile_img, loc, mask=self.black_tile_img)
                piece = np.argmax(col)
                if piece == 0:
                    img.paste(self.black_pawn_img, loc, mask=self.black_pawn_img)
                elif piece == 1:
                    img.paste(self.white_pawn_img, loc, mask=self.white_pawn_img)
        return img


def save_gif(
    images: List[Image.Image],
    path: Optional[Path] = None,
    quality: int = 40,
    duration: int = 40,
):
    if not path:
        path = next_valid_fpath("gifs/", "gif")
    assert len(images) >= 1
    images[0].save(
        str(path),
        save_all=True,
        append_images=images[1:],
        optimize=True,
        quality=quality,
        duration=duration,
    )
    logger.info("gif saved to " + str(path))


def next_valid_fpath(root_dir: Union[Path, str], suffix: str) -> Path:
    root_dir = Path(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)
    counter = 0
    while next_path := (root_dir / f"{counter}.{suffix}"):
        if next_path.exists():
            counter += 1
        else:
            return next_path
    raise RuntimeError("Shouldn't reach here")


def convert_tree_to_graph(
    tree: mctx.Tree,
    action_labels: Optional[Sequence[str]] = None,
    batch_index: int = 0,
    show_only_expanded: bool = True,
    image_path: Optional[str] = None,
    image_suffix: str = "png",
) -> pygraphviz.AGraph:
    """Converts a search tree into a Graphviz graph.
    Args:
      tree: A `Tree` containing a batch of search data.
      action_labels: Optional labels for edges, defaults to the action index.
      batch_index: Index of the batch element to plot.
    Returns:
      A Graphviz graph representation of `tree`.

    Copy-pasted from mctx library examples.
    https://github.com/deepmind/mctx/blob/main/examples/visualization_demo.py
    """
    chex.assert_rank(tree.node_values, 2)
    batch_size = tree.node_values.shape[0]
    if action_labels is None:
        action_labels = list(map(str, range(tree.num_actions)))
    elif len(action_labels) != tree.num_actions:
        raise ValueError(
            f"action_labels {action_labels} has the wrong number of actions "
            f"({len(action_labels)}). "
            f"Expecting {tree.num_actions}."
        )

    def node_to_str(node_i, reward=0, discount=1):
        return (
            f"I: {node_i}\n"
            f"R: {reward:.2f}\n"
            # f"d: {discount:.2f}\n"
            f"V: {tree.node_values[batch_index, node_i]:.2f}\n"
            f"N: {tree.node_visits[batch_index, node_i]}\n"
        )

    def edge_to_str(node_i, a_i):
        node_index = jnp.full([batch_size], node_i)
        probs = jax.nn.softmax(tree.children_prior_logits[batch_index, node_i])
        return (
            f"A: {action_labels[a_i]}\n"
            f"Q: {tree.qvalues(node_index)[batch_index, a_i]:.2f}\n"
            f"p: {probs[a_i]:.2f}\n"
        )

    graph = pygraphviz.AGraph(directed=True, imagepath=image_path)
    graph.node_attr.update(
        shape="box", height=2.5, imagepos="tc", labelloc="b", fontname="Courier New"
    )

    # Add root
    graph.add_node(
        0,
        label=node_to_str(node_i=0),
        **({"image": f"0.{image_suffix}"} if image_path else {}),
    )
    # Add all other nodes and connect them up.
    for node_id in range(tree.num_simulations):
        for action_id in range(tree.num_actions):
            # Index of children, or -1 if not expanded
            child_id = tree.children_index[batch_index, node_id, action_id]
            if show_only_expanded:
                to_show = child_id >= 0
            else:
                to_show = True
            if to_show:
                graph.add_node(
                    child_id,
                    label=node_to_str(
                        node_i=child_id,
                        reward=tree.children_rewards[batch_index, node_id, action_id],
                        discount=tree.children_discounts[
                            batch_index, node_id, action_id
                        ],
                    ),
                    **{"image": f"{child_id}.{image_suffix}"} if image_path else {},
                )
                graph.add_edge(node_id, child_id, label=edge_to_str(node_id, action_id))

    return graph


def save_state_as_image(
    vis: Visualizer,
    game_state: pyspiel.State,
    i: int,
    image_root: Union[PathLike, str],
):
    if game_state:
        target_shape = game_state.get_game().observation_tensor_shape()
        frame = np.array(game_state.observation_tensor(BASE_PLAYER))
        frame = np.moveaxis(frame.reshape(target_shape), 0, -1)
        img = vis.make_image(frame)
        img = ImageOps.contain(img, (120, 120), Image.Resampling.LANCZOS)
    else:
        img = Image.new("RGBA", (120, 120), color="black")
    with open(Path(image_root) / f"{i}.png", "wb") as f:
        img.save(f, format="PNG")


def align_game_states(
    root_state: pyspiel.State,
    search_tree: mctx.Tree,
) -> Dict[int, Optional[pyspiel.State]]:
    node_states = {0: root_state}
    for node_id in tqdm(range(search_tree.num_simulations)):
        for action_id in range(search_tree.num_actions):
            child_id = int(search_tree.children_index[0, node_id, action_id])
            if child_id < 0:
                continue
            if node_states[node_id] and (action_id != 0):
                new_state = node_states[node_id].clone()
                env_action = action_id - 1
                if env_action in new_state.legal_actions():
                    new_state.apply_action(env_action)
                    node_states[child_id] = new_state
                else:
                    node_states[child_id] = None
            else:
                node_states[child_id] = None
    return node_states
