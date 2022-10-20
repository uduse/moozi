import numpy as np
import pytest
import jax

from moozi.core.utils import (
    make_one_hot_planes,
    make_frame_planes,
    fifo_append,
    push_and_rotate_out_planes,
)
from moozi.core.scalar_transform import make_scalar_transform


@pytest.mark.parametrize("contract", [True, False])
@pytest.mark.parametrize(
    "support_min,support_max",
    [
        (-10, 10),
        (0, 30),
        (-1, 1),
    ],
)
def test_scalar_transform(support_min, support_max, contract):
    scalar_transform = make_scalar_transform(
        support_min=support_min,
        support_max=support_max,
        contract=contract,
    )
    scalars = np.random.uniform(
        low=scalar_transform.scalar_min,
        high=scalar_transform.scalar_max,
        size=(100,),
    )
    scalar_probs = scalar_transform.transform(scalars)
    outputs = scalar_transform.inverse_transform(scalar_probs)
    np.testing.assert_allclose(scalars, outputs, rtol=0.05, atol=0.1)


def test_make_action_planes():
    action_planes = make_one_hot_planes(np.array([0, 1, 2]), 2, 2, 3)
    np.testing.assert_almost_equal(action_planes[0, 0, :3], [1 / 3, 0, 0])
    np.testing.assert_almost_equal(action_planes[0, 0, 3:6], [0, 1 / 3, 0])
    np.testing.assert_almost_equal(action_planes[0, 0, 6:9], [0, 0, 1 / 3])


@pytest.mark.parametrize("use_jit", [True, False], ids=["no_jit", "jit"])
def test_make_frame_planes(use_jit):
    # num_stacked_frames = 2
    orig = np.arange(2 * 3 * 4 * 5).reshape((2, 3, 4, 5))
    if use_jit:
        new = jax.jit(make_frame_planes)(orig)
    else:
        new = make_frame_planes(orig)
    np.testing.assert_array_equal(new[1, 1, :5], orig[0, 1, 1, :])
    np.testing.assert_array_equal(new[1, 1, 5:], orig[1, 1, 1, :])


def test_push_and_rotate_out_planes():
    planes = np.arange(2 * 3).reshape((1, 2, 3))
    print(planes)
    new_plane = np.array([10, 11]).reshape((1, 2, 1))
    new_planes = push_and_rotate_out_planes(planes, new_plane)
    np.testing.assert_equal(
        new_planes,
        np.array([[[1, 2, 10], [4, 5, 11]]]),
    )


def test_push_and_rotate_out():
    old_actions = np.array([0, 1, 2, 3])
    new_action = np.array(4)
    expected_actions = np.array([1, 2, 3, 4])
    np.testing.assert_equal(
        fifo_append(old_actions, new_action), expected_actions
    )

    old_frames = np.array([0, 1, 2, 3]).reshape((4, 1, 1, 1))
    new_frame = np.array(4).reshape((1, 1, 1))
    expected_frames = np.array([1, 2, 3, 4]).reshape((4, 1, 1, 1))
    np.testing.assert_equal(fifo_append(old_frames, new_frame), expected_frames)
