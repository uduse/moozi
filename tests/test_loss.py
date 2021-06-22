import pytest
import jax.numpy as jnp
import numpy as np
import moozi as mz
from acme.jax.utils import add_batch_dim


@pytest.fixture
def train_target():
    return add_batch_dim(
        mz.replay.TrainTarget(
            stacked_frames=jnp.ones((5, 5)),
            action=[-1],
            value=[0, 0],
            last_reward=[0, 0],
            child_visits=[np.zeros(3), np.zeros(3)],
        ).cast()
    )


def test_mcts_loss(network, params, train_target):
    loss = mz.loss.MCTSLoss(num_unroll_steps=2)
    assert loss(network, params, train_target)
