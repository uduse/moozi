import contextlib
from typing import Set
import jax
from loguru import logger


def make_tape(seed: int = 0):
    tape = {}
    tape["random_key"] = jax.random.PRNGKey(seed)
    tape["output_buffer"] = tuple()
    tape["output"] = None
    return tape


# TODO: don't use context manager
@contextlib.contextmanager
def exclude(tape: dict, to_exclude: Set[str]):
    masked = {k: v for k, v in tape.items() if k not in to_exclude}
    yield masked


@contextlib.contextmanager
def include(tape: dict, to_include: Set[str]):
    if not all(k in tape for k in to_include):
        logger.warning(
            f"{tape.keys()} does not contain key {to_include - set(tape.keys())}"
        )
    masked = {k: v for k, v in tape.items() if k in to_include}
    yield masked
