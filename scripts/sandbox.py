# %%
import functools
from dataclasses import dataclass
from typing import NamedTuple

import chex
import graphviz
import haiku as hk
import jax
import jax.numpy as jnp
import tree
from acme.jax.utils import add_batch_dim, squeeze_batch_dim
from moozi.nn import (
    InitialInferenceFeatures,
    NeuralNetwork,
    # NeuralNetworkSpec,
    NNOutput,
    RecurrentInferenceFeatures,
)


# %%
def info(structure):
    print(tree.map_structure(lambda x: (x.shape, x.dtype), structure))


# %%
class NeuralNetworkSpec(NamedTuple):
    stacked_frames_shape: tuple
    dim_repr: int
    dim_action: int


class ConvBlock(hk.Module):
    def __call__(self, x):
        x = hk.Conv2D(16, (3, 3), padding="same")(x)
        x = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.9)(
            x, is_training=True
        )
        x = jax.nn.relu(x)
        return x


@dataclass
class ResBlock(hk.Module):
    output_channels: int = 16

    def __call__(self, x):
        orig_x = x
        x = hk.Conv2D(
            output_channels=self.output_channels, kernel_shape=(3, 3), padding="same"
        )(x)
        x = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.9)(
            x, is_training=True
        )
        x = jax.nn.relu(x)
        x = hk.Conv2D(
            output_channels=self.output_channels, kernel_shape=(3, 3), padding="same"
        )(x)
        x = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.9)(
            x, is_training=True
        )
        x = x + orig_x
        x = jax.nn.relu(x)
        return x


@dataclass
class ResTower(hk.Module):
    num_blocks: int

    def __call__(self, x):
        for _ in range(self.num_blocks):
            x = ResBlock()(x)
        return x


@dataclass
class ResTowerV2(hk.Module):
    num_blocks: int
    res_channels: int
    output_channels: int

    def __call__(self, x):
        for _ in range(self.num_blocks):
            x = ResBlock(output_channels=self.res_channels)(x)
        x = hk.Conv2D(
            output_channels=self.output_channels, kernel_shape=(3, 3), padding="same"
        )(x)
        return x


class MuZeroNet(hk.Module):
    def __init__(self, spec: NeuralNetworkSpec):
        super().__init__()
        self.spec = spec

    def repr_net(self, stacked_frames, dim_repr):
        # (batch_size, num_frames, height, width, channels)
        chex.assert_rank(stacked_frames, 5)
        stacked_frames = stacked_frames.transpose(0, 2, 3, 4, 1)
        stacked_frames = stacked_frames.reshape(stacked_frames.shape[:-2] + (-1,))

        hidden_state = ConvBlock()(stacked_frames)
        hidden_state = ResTower(num_blocks=20)(hidden_state)
        hidden_state = hk.Conv2D(
            output_channels=dim_repr, kernel_shape=(3, 3), padding="same"
        )(hidden_state)

        chex.assert_rank(hidden_state, 4)  # (batch_size, height, width, dim_repr)

        return hidden_state

    def pred_net(self, hidden_state):
        pred_trunk = ResTower(num_blocks=20)(hidden_state)
        pred_trunk = hk.Conv2D(
            output_channels=dim_repr, kernel_shape=(3, 3), padding="same"
        )(pred_trunk)

        chex.assert_rank(pred_trunk, 4)  # (batch_size, height, width, dim_repr)

        pred_trunk_flat = pred_trunk.reshape((pred_trunk.shape[0], -1))
        chex.assert_rank(pred_trunk_flat, 2)  # (batch_size, height * width * dim_repr)

        value = hk.Linear(output_size=1, name="pred_v")(pred_trunk_flat)
        chex.assert_shape(value, (None, 1))  # (batch_size, 1)

        policy_logits = hk.Linear(output_size=self.spec.dim_action, name="pred_p")(
            pred_trunk_flat
        )
        chex.assert_shape(
            policy_logits, (None, self.spec.dim_action)
        )  # (batch_size, dim_action)

        return value, policy_logits

    def dyna_net(self, hidden_state, action):
        action_one_hot = jax.nn.one_hot(action, num_classes=self.spec.dim_action)
        action_one_hot = action_one_hot.tile(hidden_state.shape[0:3] + (1,))

        chex.assert_equal_rank([hidden_state, action_one_hot])
        state_action_repr = jnp.concatenate((hidden_state, action_one_hot), axis=-1)

        dyna_trunk = ResTowerV2(
            num_blocks=20,
            res_channels=state_action_repr.shape[-1],
            output_channels=self.spec.dim_repr,
        )(state_action_repr)

        next_hidden_state = ResTowerV2(
            num_blocks=20,
            res_channels=self.spec.dim_repr,
            output_channels=self.spec.dim_repr,
        )(dyna_trunk)

        reward = hk.Linear(output_size=1, name="dyna_reward")(
            dyna_trunk.reshape((dyna_trunk.shape[0], -1))
        )
        chex.assert_shape(reward, (None, 1))  # (batch_size, 1)

        return next_hidden_state, reward

    def initial_inference(self, init_inf_feats: InitialInferenceFeatures):
        hidden_state = self.repr_net(init_inf_feats.stacked_frames, self.spec.dim_repr)
        value, policy_logits = self.pred_net(hidden_state)
        reward = jnp.zeros_like(value)

        chex.assert_rank([value, reward, policy_logits, hidden_state], [2, 2, 2, 4])

        return NNOutput(
            value=value,
            reward=reward,
            policy_logits=policy_logits,
            hidden_state=hidden_state,
        )

    def recurrent_inference(self, recurr_inf_feats: RecurrentInferenceFeatures):
        next_hidden_state, reward = self.dyna_net(
            recurr_inf_feats.hidden_state, recurr_inf_feats.action
        )
        value, policy_logits = self.pred_net(next_hidden_state)
        chex.assert_rank(
            [value, reward, policy_logits, next_hidden_state], [2, 2, 2, 4]
        )
        return NNOutput(
            value=value,
            reward=reward,
            policy_logits=policy_logits,
            hidden_state=next_hidden_state,
        )


def get_network(spec: NeuralNetworkSpec):
    # TODO: rename to build network
    nn = functools.partial(MuZeroNet, spec)

    init_inf_pass = hk.without_apply_rng(
        hk.transform_with_state(
            lambda init_inf_features: nn().initial_inference(init_inf_features)
        )
    )
    recurr_inf_pass = hk.without_apply_rng(
        hk.transform_with_state(
            lambda recurr_inf_features: nn().recurrent_inference(recurr_inf_features)
        )
    )

    def init(random_key):
        key_1, key_2 = jax.random.split(random_key)
        dummy_batch_dim = 1
        init_inf_params, init_inf_state = init_inf_pass.init(
            key_1,
            InitialInferenceFeatures(
                stacked_frames=jnp.ones((dummy_batch_dim,) + spec.stacked_frames_shape),
                player=jnp.array(0),
            ),
        )

        recurr_inf_params, recurr_inf_state = recurr_inf_pass.init(
            key_2,
            RecurrentInferenceFeatures(
                hidden_state=jnp.ones(
                    (dummy_batch_dim, *spec.stacked_frames_shape[1:3], spec.dim_repr)
                ),
                action=jnp.ones((dummy_batch_dim,)),
            ),
        )

        params = hk.data_structures.merge(
            init_inf_params,
            recurr_inf_params,
        )
        state = hk.data_structures.merge(
            init_inf_state,
            recurr_inf_state,
        )
        return params, state

    def _initial_inference_unbatched(params, state, init_inf_features):
        nn_out, new_state = init_inf_pass.apply(
            params, state, add_batch_dim(init_inf_features)
        )
        return squeeze_batch_dim(nn_out), new_state

    def _recurrent_inference_unbatched(params, state, recurr_inf_features):
        nn_out, new_state = recurr_inf_pass.apply(
            params, state, add_batch_dim(recurr_inf_features)
        )
        return squeeze_batch_dim(nn_out), new_state

    return NeuralNetwork(
        init,
        init_inf_pass.apply,
        recurr_inf_pass.apply,
        _initial_inference_unbatched,
        _recurrent_inference_unbatched,
    )


# %%
num_stacked_frames = 4
frame_shape = (5, 5, 3)
dim_repr = 16
dim_action = 7
stacked_frames_shape = (num_stacked_frames,) + frame_shape
spec = NeuralNetworkSpec(
    stacked_frames_shape=stacked_frames_shape, dim_repr=dim_repr, dim_action=dim_action
)
nn = get_network(spec)
rng = jax.random.PRNGKey(0)
params, state = nn.init(rng)

# %%
init_inf_feat = InitialInferenceFeatures(
    stacked_frames=jnp.ones(stacked_frames_shape), player=jnp.array(0)
)
nn_out, new_state = nn.initial_inference_unbatched(params, state, init_inf_feat)

# %%
tree.map_structure(lambda x: x.shape, new_state)

# %%
ini_inf = hk.experimental.to_dot(nn.initial_inference)(params, state, init_inf_feat)
ini_inf = graphviz.Source(ini_inf)

# %%
feat = RecurrentInferenceFeatures(hidden_state=state, action=jnp.ones((1,)))
dot = hk.experimental.to_dot(nn.recurrent_inference)(params, state, feat)
dot = graphviz.Source(dot)

# %%


# %%
import pickle

with open("/tmp/params.pkl", "wb") as f:
    pickle.dump(params, f)
# %%
