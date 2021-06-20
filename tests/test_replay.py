import tree
import moozi as mz
import numpy as np
import pytest
from numpy.testing import assert_allclose

SAMPLE = mz.replay.ReplaySample(
    frame=[[11], [22], [33], [44], [55], [66]],
    reward=[0, 200, 300, 400, 500, 600],
    is_first=[True, False, False, False, False, False],
    is_last=[False, False, False, False, False, True],
    action=[101, 102, 103, 104, 105, -1],
    root_value=[10, 20, 30, 40, 50, -1],
    child_visits=[np.arange(i, i + 3) / sum(list(range(i, i + 3))) for i in range(5)]
    + [np.array([0, 0, 0])],
).cast()

TEST_CASES = []

TEST_CASES.append(
    dict(
        name="test 1",
        sample=SAMPLE,
        start_idx=0,
        discount=0.5,
        num_unroll_steps=1,
        num_td_steps=1,
        num_stacked_frames=1,
        expected_target=mz.replay.TrainTarget(
            frame=[[11]],
            action=[101],
            value=[200 + 20 * 0.5, 300 + 30 * 0.5],
            last_reward=[0, 200],
            child_visits=SAMPLE.child_visits[0:2],
        ).cast(),
    )
)

TEST_CASES.append(
    dict(
        name="test 2",
        sample=SAMPLE,
        start_idx=0,
        discount=0.5,
        num_unroll_steps=2,
        num_td_steps=2,
        num_stacked_frames=2,
        expected_target=mz.replay.TrainTarget(
            frame=[[0], [11]],
            action=[101, 102],
            value=[
                200 + 300 * 0.5 + 30 * 0.5 ** 2,
                300 + 400 * 0.5 + 40 * 0.5 ** 2,
                400 + 500 * 0.5 + 50 * 0.5 ** 2,
            ],
            last_reward=[0, 200, 300],
            child_visits=SAMPLE.child_visits[0:3],
        ).cast(),
    )
)

TEST_CASES.append(
    dict(
        name="test 3",
        sample=SAMPLE,
        start_idx=2,
        discount=0.5,
        num_unroll_steps=1,
        num_td_steps=1,
        num_stacked_frames=1,
        expected_target=mz.replay.TrainTarget(
            frame=[[33]],
            action=[103],
            value=[400 + 40 * 0.5, 500 + 50 * 0.5],
            last_reward=[0, 400],
            child_visits=SAMPLE.child_visits[2:4],
        ).cast(),
    ),
)

TEST_CASES.append(
    dict(
        name="test 4",
        sample=SAMPLE,
        start_idx=5,
        discount=0.5,
        num_unroll_steps=1,
        num_td_steps=1,
        num_stacked_frames=3,
        expected_target=mz.replay.TrainTarget(
            frame=[[44], [55], [66]],
            action=[-1],
            value=[0, 0],
            last_reward=[0, 0],
            child_visits=[np.zeros(3), np.zeros(3)],
        ).cast(),
    )
)

TEST_CASES.append(
    dict(
        name="test 5",
        sample=SAMPLE,
        start_idx=4,
        discount=0.5,
        num_unroll_steps=3,
        num_td_steps=100,
        num_stacked_frames=3,
        expected_target=mz.replay.TrainTarget(
            frame=[[33], [44], [55]],
            action=[105, -1, -1],
            value=[600, 0, 0, 0],
            last_reward=[0, 600, 0, 0],
            child_visits=[
                SAMPLE.child_visits[4],
                np.zeros(3),
                np.zeros(3),
                np.zeros(3),
            ],
        ).cast(),
    )
)

names = list(map(lambda x: x["name"], TEST_CASES))
assert len(names) == len(set(names)), "Duplicate test cases name."


@pytest.mark.kwparametrize(*TEST_CASES)
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
    computed_target = mz.replay.make_target(
        sample,
        start_idx,
        discount,
        num_unroll_steps,
        num_td_steps,
        num_stacked_frames,
    )

    assert computed_target.value.shape[0] == computed_target.child_visits.shape[0]
    assert computed_target.value.shape[0] == computed_target.last_reward.shape[0]
    assert computed_target.value.shape[0] == computed_target.action.shape[0] + 1
    assert computed_target.frame.shape[0] == num_stacked_frames

    def _assert_shape(path, computed, expected):
        assert computed.shape == expected.shape

    def _assert_content(path, computed, expected):
        assert_allclose(computed, expected)

    tree.assert_same_structure(computed_target, expected_target)
    tree.map_structure_with_path(_assert_shape, computed_target, expected_target)
    tree.map_structure_with_path(_assert_content, computed_target, expected_target)
