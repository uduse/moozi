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
    NeuralNetworkSpec,
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
        x = hk.Conv2D(256, (3, 3), padding="same")(x)
        x = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.9)(
            x, is_training=True
        )
        x = jax.nn.relu(x)
        return x


class ResBlock(hk.Module):
    def __call__(self, x):
        orig_x = x
        x = hk.Conv2D(output_channels=256, kernel_shape=(3, 3), padding="same")(x)
        x = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.9)(
            x, is_training=True
        )
        x = jax.nn.relu(x)
        x = hk.Conv2D(output_channels=256, kernel_shape=(3, 3), padding="same")(x)
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


class MuZeroNet(hk.Module):
    def __init__(self, spec: NeuralNetworkSpec):
        super().__init__()
        self.spec = spec

    def repr_net(self, stacked_frames, dim_repr):
        x = ConvBlock()(stacked_frames)
        x = ResTower(num_blocks=2)(x)
        x = hk.Conv2D(output_channels=dim_repr, kernel_shape=(3, 3), padding="same")(x)
        return x

    def pred_net(self, hidden_state):
        pred_trunk = hk.nets.MLP(
            output_sizes=self.spec.pred_net_sizes,
            name="pred_trunk",
            activation=jnp.tanh,
            activate_final=True,
        )
        v_branch = hk.Linear(output_size=1, name="pred_v")
        p_branch = hk.Linear(output_size=self.spec.dim_action, name="pred_p")

        pred_trunk_out = pred_trunk(hidden_state)
        value = jnp.squeeze(v_branch(pred_trunk_out), axis=-1)
        policy_logits = p_branch(pred_trunk_out)
        return value, policy_logits

    def dyna_net(self, hidden_state, action):
        dyna_trunk = hk.nets.MLP(
            output_sizes=self.spec.dyna_net_sizes,
            name="dyna_trunk",
            activation=jnp.tanh,
            activate_final=True,
        )
        trans_branch = hk.nets.MLP(
            output_sizes=[self.spec.dim_repr],
            name="dyna_trans",
            activation=jnp.tanh,
            activate_final=True,
        )
        reward_branch = hk.nets.MLP(
            output_sizes=[1],
            name="dyna_reward",
            activation=jnp.tanh,
            activate_final=True,
        )

        action_one_hot = jax.nn.one_hot(action, num_classes=self.spec.dim_action)
        chex.assert_equal_rank([hidden_state, action_one_hot])
        state_action_repr = jnp.concatenate((hidden_state, action_one_hot), axis=-1)
        dyna_trunk_out = dyna_trunk(state_action_repr)
        next_hidden_states = trans_branch(dyna_trunk_out)
        next_rewards = jnp.squeeze(reward_branch(dyna_trunk_out), axis=-1)
        return next_hidden_states, next_rewards

    def initial_inference(self, init_inf_features: InitialInferenceFeatures):
        x = self.repr_net(init_inf_features.stacked_frames, self.spec.dim_repr)
        return NNOutput(
            value=jnp.zeros((1, 1)),
            reward=jnp.zeros((1, 1)),
            policy_logits=jnp.zeros((1, self.spec.dim_action)),
            hidden_state=jnp.zeros((1, self.spec.dim_repr)),
        )
        # chex.assert_rank(init_inf_features.stacked_frames, 3)
        # hidden_state = self.repr_net(init_inf_features.stacked_frames)
        # value, policy_logits = self.pred_net(hidden_state)
        # reward = jnp.zeros_like(value)
        # chex.assert_rank([value, reward, policy_logits, hidden_state], [1, 1, 2, 2])
        # return NNOutput(
        #     value=value,
        #     reward=reward,
        #     policy_logits=policy_logits,
        #     hidden_state=hidden_state,
        # )

    def recurrent_inference(self, recurr_inf_features: RecurrentInferenceFeatures):
        pass
        # # TODO: a batch-jit that infers K times?
        # next_hidden_state, reward = self.dyna_net(
        #     recurr_inf_features.hidden_state, recurr_inf_features.action
        # )
        # value, policy_logits = self.pred_net(next_hidden_state)
        # chex.assert_rank(
        #     [value, reward, policy_logits, next_hidden_state], [1, 1, 2, 2]
        # )
        # return NNOutput(
        #     value=value,
        #     reward=reward,
        #     policy_logits=policy_logits,
        #     hidden_state=next_hidden_state,
        # )


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
        batch_size = 1
        init_inf_params, _ = init_inf_pass.init(
            key_1,
            InitialInferenceFeatures(
                stacked_frames=jnp.ones((batch_size,) + spec.stacked_frames_shape),
                player=jnp.array(0),
            ),
        )

        recurr_inf_params, _ = recurr_inf_pass.init(
            key_2,
            RecurrentInferenceFeatures(
                hidden_state=jnp.ones((batch_size, spec.dim_repr)),
                action=jnp.ones((batch_size,)),
            ),
        )

        params = hk.data_structures.merge(
            init_inf_params,
            recurr_inf_params,
        )
        return params

    def _initial_inference_unbatched(params, init_inf_features):
        return squeeze_batch_dim(
            init_inf_pass.apply(params, add_batch_dim(init_inf_features))
        )

    def _recurrent_inference_unbatched(params, recurr_inf_features):
        return squeeze_batch_dim(
            recurr_inf_pass.apply(params, add_batch_dim(recurr_inf_features))
        )

    return NeuralNetwork(
        init,
        init_inf_pass.apply,
        recurr_inf_pass.apply,
        _initial_inference_unbatched,
        _recurrent_inference_unbatched,
    )


# %%
spec = NeuralNetworkSpec(stacked_frames_shape=(3, 3, 3 * 2), dim_repr=3, dim_action=9)
nn = get_network(spec)

# %%
rng = jax.random.PRNGKey(0)
info(nn.init(rng))

# %%
# info(params)
# info(state)

# # %%
# output, state = model.apply(params, state, obs)
# info(output)

# # %%
obs = jnp.ones((3, 3, 3 * 2))
print(
    hk.experimental.tabulate(
        nn.initial_inference, columns=["config", "input", "output", "params_bytes"]
    )(InitialInferenceFeatures(obs, 0))
)

# # %%
# dot = hk.experimental.to_dot(model.apply)(params, state, obs)
# dot = graphviz.Source(dot)
# # %%

# %%
