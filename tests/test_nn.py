import functools
from typing import Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from moozi.core import PolicyFeed
from moozi.core.env import GIIEnv, GIIVecEnv
from moozi.core.history_stacker import HistoryStacker
from moozi.core.scalar_transform import ScalarTransform, make_scalar_transform
from moozi.core.utils import add_batch_dim
from moozi.nn import (
    MLPArchitecture,
    NNModel,
    NNSpec,
    ResNetArchitecture,
    ResNetSpec,
    ResNetV2Architecture,
    ResNetV2Spec,
    RootFeatures,
    TransitionFeatures,
    make_model,
)
from moozi.nn.mlp import MLPSpec
from moozi.nn.naive import NaiveArchitecture
from moozi.nn.training import Trainer


@pytest.fixture(
    params=["OpenSpiel:catch", "MinAtar:Breakout-v4"],
    ids=["catch", "breakout"],
)
def env(request):
    env_name = request.param
    return GIIEnv.new(env_name)


# override pytest model fixture to test all architectures
@pytest.fixture(
    params=[
        (NaiveArchitecture, NNSpec),
        (MLPArchitecture, MLPSpec),
        (ResNetArchitecture, ResNetSpec),
        (ResNetV2Architecture, ResNetV2Spec),
    ],
    ids=["naive", "mlp", "resnet", "resnet_v2"],
)
def model(env: GIIEnv, request, history_length):
    arch_cls, spec_cls = request.param
    shape = env.spec.frame.shape
    return make_model(
        arch_cls,
        spec_cls(
            frame_rows=shape[0],
            frame_cols=shape[1],
            frame_channels=shape[2],
            repr_rows=shape[0],
            repr_cols=shape[1],
            repr_channels=2,
            dim_action=env.spec.dim_action,
            scalar_transform=ScalarTransform.new(-10, 10),
            num_players=env.spec.num_players,
            history_length=history_length,
        ),
    )


def params_and_state(model: NNModel, random_key) -> Tuple[hk.Params, hk.State]:
    return model.init_params_and_state(random_key)


@pytest.mark.parametrize("use_jit", [True, False], ids=["no_jit", "jit"])
def test_model_basic_inferences(
    env: GIIEnv,
    model: NNModel,
    params_and_state,
    use_jit,
    history_length,
):
    feed = env.init()
    env_out = env.step(feed)

    params, state = params_and_state
    if use_jit:
        model = model.with_jit()

    frames = np.repeat(env_out.frame[np.newaxis, ...], repeats=history_length, axis=0)
    actions = np.repeat(
        np.array(feed.action)[np.newaxis, ...], repeats=history_length, axis=0
    )
    root_feats = RootFeatures(
        frames=frames,
        actions=actions,
        to_play=np.array(0),
    )
    for is_training in [True, False]:
        out, _ = model.root_inference_unbatched(params, state, root_feats, is_training)
        assert out
        trans_feats = TransitionFeatures(
            hidden_state=out.hidden_state,
            action=np.array(0),
        )
        out, _ = model.trans_inference_unbatched(
            params, state, trans_feats, is_training
        )

        out, _ = model.root_inference(
            params, state, add_batch_dim(root_feats), is_training
        )
        assert out
        trans_feats = TransitionFeatures(
            hidden_state=out.hidden_state,
            action=add_batch_dim(np.array(0)),
        )
        out, _ = model.trans_inference(params, state, trans_feats, is_training)


def test_trainer(model):
    Trainer.new(
        model,
        num_unroll_steps=5,
        target_update_period=1.0,
        weight_decay=2.0,
        consistency_loss_coef=3.0,
        value_loss_coef=4.0,
        lr=5.0,
        clip_gradient=6.0,
    )
