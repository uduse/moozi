import tree
import numpy as np
import pytest
from numpy.testing import assert_allclose

from moozi import TrajectorySample, TrainTarget
from moozi.replay import make_target_from_traj

# random probs for valid actions
_ACTION_PROBS = np.random.randn(5, 3)
_ACTION_PROBS = _ACTION_PROBS / np.sum(_ACTION_PROBS, axis=1, keepdims=True)
# uniform probs for absorbing states
_ACTION_PROBS = np.append(_ACTION_PROBS, np.ones((1, 3)) / 3, axis=0)
SINGLE_PLAYER_SAMPLE = TrajectorySample(
    frame=np.arange(6).reshape(6, 1, 1, 1),
    last_reward=[0, 200, 300, 400, 500, 600],
    is_first=[True, False, False, False, False, False],
    is_last=[False, False, False, False, False, True],
    to_play=[0, 0, 0, 0, 0, 0],
    root_value=[10, 20, 30, 40, 50, 0],
    action_probs=_ACTION_PROBS,
    action=[101, 102, 103, 104, 105, -1],
    legal_actions_mask=np.ones((6, 3)),
    weight=np.ones(6),
).cast()

REPLAY_TEST_CASES = []

REPLAY_TEST_CASES.append(
    dict(
        name="test 1",
        sample=SINGLE_PLAYER_SAMPLE,
        start_idx=0,
        discount=0.5,
        num_unroll_steps=1,
        num_td_steps=1,
        num_stacked_frames=1,
        expected_target=TrainTarget(
            stacked_frames=np.array([0]).reshape(1, 1, 1),
            action=[101],
            n_step_return=[200 + 20 * 0.5, 300 + 30 * 0.5],
            last_reward=[0, 200],
            action_probs=SINGLE_PLAYER_SAMPLE.action_probs[0:2],
            root_value=[10, 20],
            weight=[1.0],
        ).cast(),
    )
)

REPLAY_TEST_CASES.append(
    dict(
        name="test 2",
        sample=SINGLE_PLAYER_SAMPLE,
        start_idx=0,
        discount=0.5,
        num_unroll_steps=2,
        num_td_steps=2,
        num_stacked_frames=2,
        expected_target=TrainTarget(
            stacked_frames=np.array([0, 0]).reshape(1, 1, 2),
            action=[101, 102],
            n_step_return=[
                200 + 300 * 0.5 + 30 * 0.5 ** 2,
                300 + 400 * 0.5 + 40 * 0.5 ** 2,
                400 + 500 * 0.5 + 50 * 0.5 ** 2,
            ],
            last_reward=[0, 200, 300],
            action_probs=SINGLE_PLAYER_SAMPLE.action_probs[0:3],
            root_value=[10, 20, 30],
            weight=[1.0],
        ).cast(),
    )
)

REPLAY_TEST_CASES.append(
    dict(
        name="test 3",
        sample=SINGLE_PLAYER_SAMPLE,
        start_idx=2,
        discount=0.5,
        num_unroll_steps=1,
        num_td_steps=1,
        num_stacked_frames=1,
        expected_target=TrainTarget(
            stacked_frames=np.array([2]).reshape(1, 1, 1),
            action=[103],
            n_step_return=[400 + 40 * 0.5, 500 + 50 * 0.5],
            last_reward=[0, 400],
            action_probs=SINGLE_PLAYER_SAMPLE.action_probs[2:4],
            root_value=[30, 40],
            weight=[1.0],
        ).cast(),
    ),
)

REPLAY_TEST_CASES.append(
    dict(
        name="test 4",
        sample=SINGLE_PLAYER_SAMPLE,
        start_idx=5,
        discount=0.5,
        num_unroll_steps=1,
        num_td_steps=1,
        num_stacked_frames=3,
        expected_target=TrainTarget(
            stacked_frames=np.array([3, 4, 5]).reshape(1, 1, 3),
            action=[-1],
            n_step_return=[0, 0],
            last_reward=[0, 0],
            action_probs=[np.ones(3) / 3, np.ones(3) / 3],
            root_value=[0, 0],
            weight=[1.0],
        ).cast(),
    )
)

REPLAY_TEST_CASES.append(
    dict(
        name="test 5",
        sample=SINGLE_PLAYER_SAMPLE,
        start_idx=4,
        discount=0.5,
        num_unroll_steps=3,
        num_td_steps=100,
        num_stacked_frames=3,
        expected_target=TrainTarget(
            stacked_frames=np.array([2, 3, 4]).reshape(1, 1, 3),
            action=[105, -1, -1],
            n_step_return=[600, 0, 0, 0],
            last_reward=[0, 600, 0, 0],
            action_probs=[
                SINGLE_PLAYER_SAMPLE.action_probs[4],
                np.ones(3) / 3,
                np.ones(3) / 3,
                np.ones(3) / 3,
            ],
            root_value=[50, 0, 0, 0],
            weight=[1.0],
        ).cast(),
    )
)

REPLAY_TEST_CASES.append(
    dict(
        name="test 6",
        sample=SINGLE_PLAYER_SAMPLE,
        start_idx=5,
        discount=0.5,
        num_unroll_steps=3,
        num_td_steps=100,
        num_stacked_frames=3,
        expected_target=TrainTarget(
            stacked_frames=np.array([3, 4, 5]).reshape(1, 1, 3),
            action=[-1, -1, -1],
            n_step_return=[0, 0, 0, 0],
            last_reward=[0, 0, 0, 0],
            action_probs=[
                np.ones(3) / 3,
                np.ones(3) / 3,
                np.ones(3) / 3,
                np.ones(3) / 3,
            ],
            root_value=[0, 0, 0, 0],
            weight=[1.0],
        ).cast(),
    )
)


TWO_PLAYER_SAMPLE = TrajectorySample(
    frame=np.arange(4).reshape(4, 1, 1, 1),
    last_reward=[0, 200, -300, 400],
    is_first=[True, False, False, False],
    is_last=[False, False, False, True],
    to_play=[0, 1, 0, -1],
    root_value=[10, 20, 30, 0],
    action_probs=[np.arange(i, i + 3) / sum(list(range(i, i + 3))) for i in range(3)]
    + [np.ones(3) / 3],
    action=[101, 102, 103, -1],
    legal_actions_mask=np.ones((6, 3)),
    weight=np.ones(6),
).cast()

REPLAY_TEST_CASES.append(
    dict(
        name="test 7",
        sample=TWO_PLAYER_SAMPLE,
        start_idx=0,
        discount=0.5,
        num_unroll_steps=1,
        num_td_steps=100,
        num_stacked_frames=1,
        expected_target=TrainTarget(
            stacked_frames=np.array([0]).reshape(1, 1, 1),
            action=[101],
            n_step_return=[
                200 + -300 * 0.5 + 400 * 0.5 ** 2,
                -300 + 400 * 0.5,
            ],
            last_reward=[0, 200],
            action_probs=TWO_PLAYER_SAMPLE.action_probs[0:2],
            root_value=[10, 20],
            weight=[1.0],
        ).cast(),
    )
)

REPLAY_TEST_CASES.append(
    dict(
        name="test 8",
        sample=TWO_PLAYER_SAMPLE,
        start_idx=1,
        discount=0.5,
        num_unroll_steps=2,
        num_td_steps=1,
        num_stacked_frames=2,
        expected_target=TrainTarget(
            stacked_frames=np.array([0, 1]).reshape(1, 1, 2),
            action=[102, 103],
            n_step_return=[-300 + 30 * 0.5, 400, 0],
            last_reward=[0, -300, 400],
            action_probs=TWO_PLAYER_SAMPLE.action_probs[1:4],
            root_value=[20, 30, 0],
            weight=[1.0],
        ).cast(),
    )
)

REPLAY_TEST_CASES.append(
    dict(
        name="test 9",
        sample=TWO_PLAYER_SAMPLE,
        start_idx=2,
        discount=0.5,
        num_unroll_steps=3,
        num_td_steps=2,
        num_stacked_frames=1,
        expected_target=TrainTarget(
            stacked_frames=np.array([2]).reshape(1, 1, 1),
            action=[103, -1, -1],
            n_step_return=[400, 0, 0, 0],
            last_reward=[0, 400, 0, 0],
            action_probs=[
                TWO_PLAYER_SAMPLE.action_probs[2],
                np.ones(3) / 3,
                np.ones(3) / 3,
                np.ones(3) / 3,
            ],
            root_value=[30, 0, 0, 0],
            weight=[1.0],
        ).cast(),
    )
)

names = list(map(lambda x: x["name"], REPLAY_TEST_CASES))
assert len(names) == len(set(names)), "Duplicate test cases name."


@pytest.mark.kwparametrize(*REPLAY_TEST_CASES)
def test_make_target(
    name,
    sample,
    start_idx,
    discount,
    num_unroll_steps,
    num_td_steps,
    num_stacked_frames,
    expected_target,
):
    computed_target = make_target_from_traj(
        sample,
        start_idx,
        discount,
        num_unroll_steps,
        num_td_steps,
        num_stacked_frames,
    )

    # TODO: replace with chex assertions
    assert (
        computed_target.n_step_return.shape[0] == computed_target.action_probs.shape[0]
    )
    assert (
        computed_target.n_step_return.shape[0] == computed_target.last_reward.shape[0]
    )
    assert computed_target.n_step_return.shape[0] == computed_target.action.shape[0] + 1

    def _assert_shape(path, computed, expected):
        assert computed.shape == expected.shape

    def _assert_content(path, computed, expected):
        assert_allclose(computed, expected)

    tree.assert_same_structure(computed_target, expected_target)
    tree.map_structure_with_path(_assert_shape, computed_target, expected_target)
    tree.map_structure_with_path(_assert_content, computed_target, expected_target)
