import numpy as np
from moozi.utils import get_project_root
from typing import Optional, Union, List
from PIL import Image
from pathlib import Path
from loguru import logger


class BreakthroughVisualizer:
    def __init__(self, num_rows: int, num_cols: int):
        self.num_rows = num_rows
        self.num_cols = num_cols
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

    def make_image(
        self,
        frame,
    ) -> Image:
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
