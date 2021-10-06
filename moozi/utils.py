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


def get_uuid():
    return uuid.uuid4().hex[:8]


def convert_to_anytree(policy_tree_root, anytree_root=None, action="_"):
    anytree_child = anytree.Node(
        id=get_uuid(),
        name=action,
        parent=anytree_root,
        prior=policy_tree_root.prior,
        reward=np.round(np.array(policy_tree_root.network_output.reward).item(), 3),
        value=np.round(np.array(policy_tree_root.network_output.value).item(), 3),
    )
    for next_action, policy_tree_child in policy_tree_root.children:
        convert_to_anytree(policy_tree_child, anytree_child, next_action)
    return anytree_child


def nodeattrfunc(node):
    return f'label="value: {node.value:.3f}"'


def edgeattrfunc(parent, child):
    return f'label="action: {child.name} \nprob: {child.prior:.3f}\nreward: {child.reward:.3f}"'


_partial_exporter = functools.partial(
    UniqueDotExporter,
    nodenamefunc=lambda node: node.id,
    nodeattrfunc=nodeattrfunc,
    edgeattrfunc=edgeattrfunc,
)


def anytree_to_png(anytree_root, file_path):
    _partial_exporter(anytree_root).to_picture(file_path)
