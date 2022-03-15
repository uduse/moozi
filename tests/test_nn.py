import functools
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from moozi.core import PolicyFeed
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


@pytest.fixture(
    scope="module",
    params=[
        (NaiveArchitecture, NNSpec),
        (MLPArchitecture, MLPSpec),
        (ResNetArchitecture, ResNetSpec),
    ],
    ids=["naive", "mlp", "resnet"],
)
def model(request):
    arch_cls, spec_cls = request.param
    return make_model(arch_cls, spec_cls())


def _test_model_inference(model, params, state, policy_feed):
    assert model.spec.stacked_frames_shape == policy_feed.stacked_frames.shape
    root_feats = RootFeatures(
        stacked_frames=policy_feed.stacked_frames, player=np.array(policy_feed.to_play)
    )
    is_training = False
    out, _ = model.root_inference_unbatched(params, state, root_feats, is_training)
    assert out
    trans_feats = TransitionFeatures(hidden_state=out.hidden_state, action=np.array(0))
    out, _ = model.trans_inference_unbatched(params, state, trans_feats, is_training)
    assert out.hidden_state.shape[-1] == model.spec.dim_repr


def test_model_basic_inferences(model: NNModel, params, state, policy_feed: PolicyFeed):
    _test_model_inference(model, params, state, policy_feed)


def test_model_jit_inferences(model: NNModel, params, state, policy_feed: PolicyFeed):
    model = model.with_jit()
    _test_model_inference(model, params, state, policy_feed)


def test_resnet(num_stacked_frames):
    frame_shape = (2, 3, 4)
    dim_repr = 5
    dim_action = 6

    stacked_frames_shape = frame_shape[:-1] + (num_stacked_frames * frame_shape[-1],)
    spec = ResNetSpec(
        stacked_frames_shape=stacked_frames_shape,
        dim_repr=dim_repr,
        dim_action=dim_action,
    )
    nn = make_model(ResNetArchitecture, spec)
    rng = jax.random.PRNGKey(0)
    params, state = nn.init_params_and_state(rng)
    root_inf_feats = RootFeatures(
        stacked_frames=jnp.ones(stacked_frames_shape), player=jnp.array(0)
    )
    nn_out, new_state = nn.root_inference_unbatched(
        params, state, root_inf_feats, is_training=False
    )
    assert nn_out
    assert new_state

    trans_inf_feats = TransitionFeatures(hidden_state=nn_out.hidden_state, action=0)
    nn_out, new_state = nn.trans_inference_unbatched(
        params, state, trans_inf_feats, is_training=False
    )
    assert nn_out
    assert new_state
