import functools
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from moozi.core import PolicyFeed
from moozi.core.scalar_transform import make_scalar_transform
from moozi.nn import (
    NNSpec,
    ResNetArchitecture,
    MLPArchitecture,
    RootFeatures,
    TransitionFeatures,
    make_model,
    NNModel,
    ResNetSpec,
)
from moozi.nn.mlp import MLPSpec
from moozi.nn.naive import NaiveArchitecture


# override pytest model fixture to test all architectures
@pytest.fixture(
    params=[
        (NaiveArchitecture, NNSpec),
        (MLPArchitecture, MLPSpec),
        (ResNetArchitecture, ResNetSpec),
    ],
    ids=["naive", "mlp", "resnet"],
)
def model(env_spec, num_stacked_frames, request):
    arch_cls, spec_cls = request.param

    single_frame_shape = env_spec.observations.observation.shape
    obs_rows, obs_cols = single_frame_shape[0:2]
    obs_channels = single_frame_shape[-1] * num_stacked_frames
    dim_action = env_spec.actions.num_values

    assert issubclass(spec_cls, NNSpec)
    return make_model(
        arch_cls,
        spec_cls(
            obs_rows=obs_rows,
            obs_cols=obs_cols,
            obs_channels=obs_channels,
            repr_rows=obs_rows,
            repr_cols=obs_cols,
            repr_channels=2,
            dim_action=dim_action,
            scalar_transform=make_scalar_transform(-30, 30),
        ),
    )


@pytest.mark.parametrize("use_jit", [True, False], ids=["no_jit", "jit"])
def test_model_basic_inferences(
    model: NNModel, params, state, policy_feed: PolicyFeed, use_jit
):
    if use_jit:
        model = model.with_jit()
    root_feats = RootFeatures(
        obs=policy_feed.stacked_frames, player=np.array(policy_feed.to_play)
    )
    is_training = False
    out, _ = model.root_inference_unbatched(params, state, root_feats, is_training)
    assert out
    trans_feats = TransitionFeatures(hidden_state=out.hidden_state, action=np.array(0))
    out, _ = model.trans_inference_unbatched(params, state, trans_feats, is_training)
