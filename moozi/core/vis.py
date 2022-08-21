from multiprocessing.sharedctypes import Value
from os import PathLike
from tqdm import tqdm
from pathlib import Path
from typing import List, Optional, Sequence, Union, Dict, Tuple

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
from PIL import Image, ImageOps, ImageDraw, ImageChops, ImageFont, ImageFilter


def get_font(name="courier") -> ImageFont.ImageFont:
    try:
        font = ImageFont.truetype(f"{name}.ttf", 10)
    except:
        font_path_user_root = str(Path(f"~/.fonts/{name}.ttf").expanduser().resolve())
        font = ImageFont.truetype(font_path_user_root, 10)
    return font


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
        self.grey_tile_img = Image.open(root / "assets/grey_tile.png").convert("RGBA")
        assert (
            self.white_pawn_img.size
            == self.black_pawn_img.size
            == self.white_tile_img.size
            == self.grey_tile_img.size
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
                    img.paste(self.grey_tile_img, loc, mask=self.grey_tile_img)
                piece = np.argmax(col)
                if piece == 0:
                    img.paste(self.black_pawn_img, loc, mask=self.black_pawn_img)
                elif piece == 1:
                    img.paste(self.white_pawn_img, loc, mask=self.white_pawn_img)
                else:
                    # empty tile
                    pass
        return img

    def make_banner(self, to_play: int) -> Image:
        assert self.num_cols >= 2
        banner_width = self.token_width * self.num_cols
        banner_height = self.token_height
        im = Image.new("RGBA", (banner_width, banner_height), color="white")

        terminal = False
        if to_play == 0:
            im.paste(self.black_pawn_img, (0, 0), mask=self.black_pawn_img)
        elif to_play == 1:
            im.paste(self.white_pawn_img, (0, 0), mask=self.white_pawn_img)
        else:
            assert to_play == pyspiel.PlayerId.TERMINAL
            terminal = True

        draw = ImageDraw.Draw(im)
        if terminal:
            content = "game ends"
        else:
            content = " to play"
        draw.text((0, 0), content, fill="black", font=get_font())
        return im


def save_gif(
    images: List[Image.Image],
    path: Union[PathLike, str, None] = None,
    quality: int = 40,
    duration: int = 40,
):
    if not path:
        path = next_valid_fpath("gifs/", "gif")
    path = Path(path).expanduser()
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
    image_dir: Union[str, PathLike, None] = None,
    image_suffix: str = "png",
    extra: Optional[Dict[int, str]] = None,
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

    def node_to_str(node_i, reward=0):
        if extra and (int(node_i) in extra):
            extra_label = f"{extra[int(node_i)]}\l"
        else:
            extra_label = ""
        return (
            extra_label + f"I: {node_i}\l"
            f"R: {reward:.2f}\l"
            # f"d: {discount:.2f}\l"
            f"V: {tree.node_values[batch_index, node_i]:.2f}\l"
            f"N: {tree.node_visits[batch_index, node_i]}\l"
        )

    def edge_to_str(node_i, a_i):
        node_index = jnp.full([batch_size], node_i)
        probs = jax.nn.softmax(tree.children_prior_logits[batch_index, node_i])
        return (
            f"A: {action_labels[a_i]}\l"
            f"Q: {tree.qvalues(node_index)[batch_index, a_i]:.2f}\l"
            f"p: {probs[a_i]:.2f}\l"
        )

    image_dir = str(image_dir)
    graph = pygraphviz.AGraph(directed=True, imagepath=image_dir)
    graph.node_attr.update(
        imagescale=True,
        shape="box",
        # width=2.5,
        # height=4,
        imagepos="tc",
        fixed_size=True,
        labelloc="b",
        fontname="Courier New",
    )
    graph.edge_attr.update(fontname="Courier New")

    # Add root
    graph.add_node(
        0,
        label=node_to_str(node_i=0),
        **({"image": f"0.{image_suffix}"} if image_dir else {}),
    )
    # Add all other nodes and connect them up.
    for node_id in tqdm(range(tree.num_simulations + 1), desc="converting to graph"):
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
                    ),
                    width=2.5,
                    height=4,
                    **{"image": f"{child_id}.{image_suffix}"} if image_dir else {},
                )
                graph.add_edge(node_id, child_id, label=edge_to_str(node_id, action_id))
    return graph


def cat_images(
    images: List[Image.Image],
    max_columns: int = 1,
    border: bool = False,
):
    if border:
        images = [
            ImageOps.expand(image, border=(2, 2, 2, 2), fill="black")
            for image in images
        ]

    width, height = images[0].width, images[0].height
    num_columns = min(len(images), max_columns)
    num_rows = int(np.ceil(len(images) / max_columns))
    canvas_width = width * num_columns
    canvas_height = height * num_rows
    dst = Image.new("RGBA", (canvas_width, canvas_height))
    for row in range(num_rows):
        for col in range(num_columns):
            idx = row * num_columns + col
            if idx >= len(images):
                continue
            dst.paste(images[idx], (col * width, row * height))
    return dst


def stack_images(images: List[Image.Image]):
    assert all(im.width == images[0].width for im in images)
    canvas_size = images[0].width, sum(im.height for im in images)
    dst = Image.new("RGBA", canvas_size)
    y_ptr = 0
    for im in images:
        dst.paste(im, (0, y_ptr))
        y_ptr += im.height
    return dst


def game_state_to_image(vis: Visualizer, game_state: Optional[pyspiel.State]):
    if game_state:
        target_shape = game_state.get_game().observation_tensor_shape()
        frame = np.array(game_state.observation_tensor(BASE_PLAYER))
        frame = np.moveaxis(frame.reshape(target_shape), 0, -1)
        img = vis.make_image(frame)
        # banner = vis.make_banner(to_play=game_state.current_player())
        # img = stack_images([img, banner])
        return ImageOps.contain(img, (250, 250), Image.Resampling.LANCZOS)
    else:
        return Image.new("RGBA", (250, 250), color="white")


def convert_game_states_to_images(
    vis: Visualizer,
    game_states: Dict[int, pyspiel.State],
    image_root: Union[PathLike, str],
    parents: Optional[Dict[int, int]] = None,
) -> Dict[int, Image.Image]:
    state_images = {}
    for idx, game_state in tqdm(game_states.items(), desc='making images'):
        path = Path(image_root) / f"{idx}.png"
        state_images[idx] = game_state_to_image(vis, game_state)
    if parents is None:
        return state_images
    else:
        diff_images = {0: state_images[0]}
        for child_idx, parent_idx in parents.items():
            diff_images[child_idx] = diff_image(
                state_images[parent_idx], state_images[child_idx]
            )
        return diff_images


def align_game_states(
    root_state: pyspiel.State,
    search_tree: mctx.Tree,
) -> Dict[int, Optional[pyspiel.State]]:
    node_states = {0: root_state}
    for node_id in tqdm(
        range(search_tree.num_simulations), desc="aligning game states"
    ):
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


def diff_image(
    before: Image.Image,
    after: Image.Image,
    opacity: int = 25,
    mask_rgb: Tuple[int, int, int] = (255, 0, 127),
):
    translucent = Image.new("RGB", before.size, mask_rgb)
    diff = ImageChops.difference(before, after)
    mask = diff.convert("L").point(lambda x: opacity if x else 0)
    mask = mask.filter(ImageFilter.GaussianBlur(radius = 1))
    ret = after.copy()
    ret.paste(translucent, (0, 0), mask)
    return ret


def visualize_search_tree(
    vis: Visualizer,
    search_tree: mctx.Tree,
    root_state: pyspiel.State,
    output_dir: Union[PathLike, str],
    batch_idx: int = 0,
) -> pygraphviz.AGraph:
    output_dir = Path(output_dir).expanduser()
    game_states = align_game_states(root_state, search_tree)
    image_dir = output_dir / "imgs"
    image_dir.mkdir(parents=True, exist_ok=True)
    parents = {}
    for child_idx in range(search_tree.num_simulations + 1):
        parent_idx = int(search_tree.parents[batch_idx, child_idx])
        if parent_idx >= 0:
            parents[child_idx] = parent_idx
    game_state_images = convert_game_states_to_images(
        vis, game_states, image_dir, parents=parents
    )
    for key, image in game_state_images.items():
        image.save(image_dir / f"{key}.png")
    extra = {}
    for idx, game_state in game_states.items():
        if game_state:
            to_play = game_state.current_player()
            if to_play == 0:
                label = "black to play"
            elif to_play == 1:
                label = "white to play"
            elif to_play == pyspiel.PlayerId.TERMINAL:
                label = "game ends"
            extra[idx] = label
        else:
            extra[idx] = 'delusion'
    g = convert_tree_to_graph(search_tree, image_dir=image_dir, extra=extra)
    g.layout(prog="dot")
    g.draw(output_dir / "search_tree.png")
    g.write(output_dir / "search_tree.dot")
    logger.info("Search tree saved to " + str(output_dir))
    return g
