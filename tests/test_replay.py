import tree
import moozi as mz
import numpy as np
import pytest
from numpy.testing import assert_allclose


TEST_CASES = []

_sample = mz.replay.ReplaySample(
    frame=[[11], [22], [33], [44], [55], [66]],
    reward=[0, 200, 300, 400, 500, 600],
    is_first=[True, False, False, False, False, False],
    is_last=[False, False, False, False, False, True],
    action=[101, 102, 103, 104, 105, 106],
    root_value=[-10, -20, -30, -40, -50, -1],
    child_visits=[np.arange(i, i + 3) / sum(list(range(i, i + 3))) for i in range(6)],
).cast()

TEST_CASES.extend(
    [
        dict(
            sample=_sample,
            start_idx=0,
            discount=0.5,
            num_unroll_steps=1,
            num_td_steps=1,
            num_stacked_frames=2,
            expected_target=mz.replay.TrainTarget(
                frame=[[[0], [11]], [[11], [22]]],
                action=[101],
                value=[200 + (-20) * 0.5, 300 + (-30) * 0.5],
                last_reward=[0, 200],
                child_visits=[_sample.child_visits[0], _sample.child_visits[1]],
            ).cast(),
        )
    ]
)


@pytest.mark.kwparametrize(*TEST_CASES)
def test_make_target(
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

    def _assert_shape(path, computed, expected):
        assert computed.shape == expected.shape

    def _assert_content(path, computed, expected):
        assert_allclose(computed, expected)

    tree.assert_same_structure(computed_target, expected_target)
    tree.map_structure_with_path(_assert_shape, computed_target, expected_target)
    tree.map_structure_with_path(_assert_content, computed_target, expected_target)
