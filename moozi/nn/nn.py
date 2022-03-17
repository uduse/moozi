import copy
import functools
from dataclasses import dataclass
from typing import Callable, NamedTuple, Tuple, Union, Type
import chex

import haiku as hk
import jax
import jax.numpy as jnp
from acme.jax.utils import add_batch_dim, squeeze_batch_dim
import tree

# NOTE: NamedTuple are used for data structures that need to be passed to jax.jit functions
# TODO: could be updated using dataclass registry https://github.com/google/jax/issues/2371
# NOTE: `?` here means the batch size is optional
class NNOutput(NamedTuple):
    value: jnp.ndarray  # (batch_size?, 1)
    reward: jnp.ndarray  # (batch_size?, 1)
    policy_logits: jnp.ndarray  # (batch_size?, num_actions)
    hidden_state: jnp.ndarray  # (batch_size?, repr_rows, repr_cols, repr_channels)


class RootFeatures(NamedTuple):
    obs: jnp.ndarray  # (batch_size?, obs_rows, obs_cols, obs_channels)
    player: jnp.ndarray  # (batch_size?, 1)


class TransitionFeatures(NamedTuple):
    hidden_state: jnp.ndarray  # (batch_size?, repr_rows, repr_cols, repr_channels)
    action: jnp.ndarray  # (batch_size?, 1)


@dataclass
class NNSpec:
    obs_rows: int
    obs_cols: int
    obs_channels: int

    repr_rows: int
    repr_cols: int
    repr_channels: int

    dim_action: int


class NNArchitecture(hk.Module):
    def __init__(self, spec: NNSpec):
        """Partiallly complete neural network model that defines the basic structure of the model.

        :param spec: more specification that completes the model
        :type spec: NNSpec
        """
        super().__init__()
        self.spec = spec

    def _repr_net(self, stacked_frames, is_training):
        raise NotImplementedError

    def _pred_net(self, hidden_state, is_training):
        raise NotImplementedError

    def _dyna_net(self, hidden_state, action, is_training):
        raise NotImplementedError

    def root_inference(self, root_feats: RootFeatures, is_training: bool):
        hidden_state = self._repr_net(root_feats.obs, is_training)
        value, policy_logits = self._pred_net(hidden_state, is_training)
        reward = jnp.zeros_like(value)

        chex.assert_rank([value, reward, policy_logits, hidden_state], [2, 2, 2, 4])

        return NNOutput(
            value=value,
            reward=reward,
            policy_logits=policy_logits,
            hidden_state=hidden_state,
        )

    def trans_inference(self, trans_feats: TransitionFeatures, is_training: bool):
        next_hidden_state, reward = self._dyna_net(
            trans_feats.hidden_state,
            trans_feats.action,
            is_training,
        )
        value, policy_logits = self._pred_net(next_hidden_state, is_training)
        chex.assert_rank(
            [value, reward, policy_logits, next_hidden_state], [2, 2, 2, 4]
        )
        return NNOutput(
            value=value,
            reward=reward,
            policy_logits=policy_logits,
            hidden_state=next_hidden_state,
        )


# TODO: also add action histories as bias planes
@dataclass
class NNModel:
    """
    Complete neural network model that's ready for initiailization and inference.

    Note that inference functions could be jitted, but need to pass with
    `jax.jit(..., static_argnames="is_training")` to make `is_training` static,
    or alternatively `jax.jit(..., static_argnums=3")`.
    """

    spec: NNSpec
    init_params_and_state: Callable
    root_inference: Callable[
        [hk.Params, hk.State, RootFeatures, bool], Tuple[NNOutput, hk.State]
    ]
    trans_inference: Callable[
        [hk.Params, hk.State, TransitionFeatures, bool], Tuple[NNOutput, hk.State]
    ]
    root_inference_unbatched: Callable[
        [hk.Params, hk.State, RootFeatures, bool], Tuple[NNOutput, hk.State]
    ]
    trans_inference_unbatched: Callable[
        [hk.Params, hk.State, TransitionFeatures, bool], Tuple[NNOutput, hk.State]
    ]
    hk_transformed: hk.MultiTransformedWithState

    def with_jit(self):
        # TODO: add chex assert max trace
        return NNModel(
            spec=copy.deepcopy(self.spec),
            init_params_and_state=self.init_params_and_state,
            root_inference=jax.jit(self.root_inference, static_argnums=3),
            trans_inference=jax.jit(self.trans_inference, static_argnums=3),
            root_inference_unbatched=jax.jit(
                self.root_inference_unbatched, static_argnums=3
            ),
            trans_inference_unbatched=jax.jit(
                self.trans_inference_unbatched, static_argnums=3
            ),
            hk_transformed=self.hk_transformed,
        )


def make_model(architecture_cls: Type[NNArchitecture], spec: NNSpec) -> NNModel:
    arch = functools.partial(architecture_cls, spec)

    def multi_transform_target():
        module = arch()

        def module_walk(root_feats, trans_feats):
            root_out = module.root_inference(root_feats, is_training=True)
            trans_out = module.trans_inference(trans_feats, is_training=True)
            return (trans_out, root_out)

        return module_walk, (module.root_inference, module.trans_inference)

    hk_transformed = hk.multi_transform_with_state(multi_transform_target)

    def init_params_and_state(rng):
        batch = 1
        obs_shape = (batch, spec.obs_rows, spec.obs_cols, spec.obs_channels)
        root_feats = RootFeatures(
            obs=jnp.ones(obs_shape),
            player=jnp.ones((batch,)),
        )
        hidden_state_shape = (batch, spec.repr_rows, spec.repr_cols, spec.repr_channels)
        trans_feats = TransitionFeatures(
            hidden_state=jnp.ones(hidden_state_shape),
            action=jnp.ones((batch,)),
        )

        return hk_transformed.init(rng, root_feats, trans_feats)

    # workaround for https://github.com/deepmind/dm-haiku/issues/325#event-6253407164
    # TODO: follow up update for the issue and make changes accordingly
    # TODO: or, alternatively, actually use random key so we could use things like dropouts
    dummy_random_key = jax.random.PRNGKey(0)

    def root_inference(params, state, feats, is_training):
        return hk_transformed.apply[0](
            params, state, dummy_random_key, feats, is_training
        )

    def trans_inference(params, state, feats, is_training):
        return hk_transformed.apply[1](
            params, state, dummy_random_key, feats, is_training
        )

    def root_inference_unbatched(params, state, feats, is_training):
        out, state = root_inference(params, state, add_batch_dim(feats), is_training)
        return squeeze_batch_dim(out), state

    def trans_inference_unbatched(params, state, feats, is_training):
        out, state = trans_inference(params, state, add_batch_dim(feats), is_training)
        return squeeze_batch_dim(out), state

    return NNModel(
        spec,
        init_params_and_state,
        root_inference,
        trans_inference,
        root_inference_unbatched,
        trans_inference_unbatched,
        hk_transformed,
    )
