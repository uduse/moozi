from dataclasses import dataclass
import random
import functools
import typing
import uuid
from logging import root

import acme.jax.variable_utils
import anytree
import chex
import dm_env
import numpy as np
from anytree.exporter import DotExporter, UniqueDotExporter


class SimpleBuffer(object):
    r"""A simple FIFO queue."""

    def __init__(self, size: int = 1000) -> None:
        self._list: list = []
        self._size = size

    def put(self, value):
        self._list.append(value)
        if len(self._list) > self._size:
            self._list = self._list[-self._size :]

    def get(self):
        return self._list

    def is_full(self) -> bool:
        return len(self._list) == self._size

    def __len__(self):
        return len(self._list)

    @property
    def size(self):
        return self._size

    def clear(self):
        self._list.clear()


def print_traj_in_env(env):
    timestep = env.reset()
    while True:
        print(env.environment.environment.get_state.observation_string(0))
        actions = env.environment.get_state.legal_actions()
        action = random.choice(actions)
        print("a:", action, "\n")
        timestep = env.step([action])
        if timestep.last():
            print(env.environment.environment.get_state.observation_string(0))
            print("reward:", timestep.reward)
            break


def convert_timestep(timestep):
    return timestep._replace(observation=timestep.observation[0])


def frame_to_str_gen(frame):
    for irow, row in enumerate(frame):
        for val in row:
            if np.isclose(val, 0.0):
                yield "."
                continue
            assert np.isclose(val, 1), val
            if irow == len(frame) - 1:
                yield "X"
            else:
                yield "O"
        yield "\n"


def frame_to_str(frame):
    return "".join(frame_to_str_gen(frame))


# def get_uuid():
#     return uuid.uuid4().hex[:8]


# def convert_to_anytree(policy_tree_root, anytree_root=None, action="_"):
#     anytree_child = anytree.Node(
#         id=get_uuid(),
#         name=action,
#         parent=anytree_root,
#         prior=policy_tree_root.prior,
#         reward=np.round(np.array(policy_tree_root.network_output.reward).item(), 3),
#         value=np.round(np.array(policy_tree_root.network_output.value).item(), 3),
#     )
#     for next_action, policy_tree_child in policy_tree_root.children:
#         convert_to_anytree(policy_tree_child, anytree_child, next_action)
#     return anytree_child


# def nodeattrfunc(node):
#     return f'label="value: {node.value:.3f}"'


# def edgeattrfunc(parent, child):
#     return f'label="action: {child.name} \nprob: {child.prior:.3f}\nreward: {child.reward:.3f}"'


# _partial_exporter = functools.partial(
#     UniqueDotExporter,
#     nodenamefunc=lambda node: node.id,
#     nodeattrfunc=nodeattrfunc,
#     edgeattrfunc=edgeattrfunc,
# )


# def anytree_to_png(anytree_root, file_path):
#     _partial_exporter(anytree_root).to_picture(file_path)


def as_coroutine(func):
    import inspect

    if inspect.iscoroutinefunction(func):
        return func
    else:

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper


import contextlib
import time


@dataclass(repr=False)
class WallTimer:
    name: str = "WallTimer"
    start_time: float = 0.0
    end_time: float = 0.0
    delta: float = 0.0

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end()

    def start(self):
        self.start_time = time.time()

    def end(self):
        self.end_time = time.time()
        self.delta += self.end_time - self.start_time

    def reset(self):
        self.delta = 0.0

    def print(self):
        print(f"{self.name}: {self.delta} s")


def check_ray_gpu():
    import ray

    ray.init(ignore_reinit_error=True)

    @ray.remote(num_gpus=1)
    def use_gpu():
        import os
        import jax

        print("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
        print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
        print("jax devices:", jax.devices())

    ray.get(use_gpu.remote())
