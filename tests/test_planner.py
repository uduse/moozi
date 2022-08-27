import numpy as np
import pytest
from moozi.core.env import GIIEnv
from moozi.core.scalar_transform import make_scalar_transform
from moozi.core.utils import add_batch_dim
from moozi import GII, HistoryStacker
from moozi.nn import RootFeatures, make_model, NaiveArchitecture, NNSpec
from moozi.planner import PlannerFeed, Planner


@pytest.mark.parametrize("num_players", [1, 2])
@pytest.mark.parametrize(
    "search_type,search_kwargs",
    [
        (
            "muzero",
            dict(dirichlet_alpha=0.1, dirichlet_fraction=0.1, pb_c_init=1.75),
        ),
        (
            "gumbel_muzero",
            dict(gumbel_scale=1, max_num_considered_actions=8),
        ),
    ],
)
def test_planner(random_key, num_players, search_type, search_kwargs):
    dim_action = 3
    scalar_transform = make_scalar_transform(-3, 3)
    frame_shape = (2, 3, 4)
    frame = np.arange(np.prod(frame_shape)).reshape(frame_shape).astype(np.float32)
    action = np.array(1, dtype=np.int32)
    legal_actions = np.ones(dim_action, dtype=np.bool8)
    to_play = np.array(num_players - 1)  # last player to play
    model = make_model(
        NaiveArchitecture,
        NNSpec(
            dim_action=dim_action,
            num_players=num_players,
            history_length=1,
            frame_rows=frame_shape[0],
            frame_cols=frame_shape[1],
            frame_channels=frame_shape[2],
            repr_rows=frame_shape[0],
            repr_cols=frame_shape[1],
            repr_channels=frame_shape[2],
            scalar_transform=scalar_transform,
        ),
    )
    params, state = model.init_params_and_state(random_key)
    root_feats = RootFeatures(
        frames=add_batch_dim(frame),
        actions=add_batch_dim(action),
        to_play=add_batch_dim(to_play),
    )
    planner_feed = PlannerFeed(
        params=params,
        state=state,
        root_feats=root_feats,
        legal_actions=add_batch_dim(legal_actions),
        random_key=random_key,
    )
    planner = Planner(
        batch_size=1,
        dim_action=dim_action,
        model=model,
        num_players=num_players,
        search_type=search_type,
        kwargs=search_kwargs,
    )
    assert planner.run(planner_feed)
