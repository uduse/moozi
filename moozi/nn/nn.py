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
class NNOutput(NamedTuple):
    value: jnp.ndarray  # (batch_size?, 1)
    reward: jnp.ndarray  # (batch_size?, 1)
    policy_logits: jnp.ndarray  # (batch_size?, num_actions)
    hidden_state: jnp.ndarray  # (batch_size?, height, width, dim_repr)


class RootFeatures(NamedTuple):
    stacked_frames: jnp.ndarray  # (batch_size?, height, width, channels)
    player: jnp.ndarray  # (batch_size?, 1)


class TransitionFeatures(NamedTuple):
    hidden_state: jnp.ndarray  # (batch_size?, height, width, dim_repr)
    action: jnp.ndarray  # (batch_size?, 1)


@dataclass
class NNSpec:
    stacked_frames_shape: tuple
    dim_repr: int
    dim_action: int


class NNArchitecture(hk.Module):
    def __init__(self, spec: NNSpec):
        """Partiallly complete neural network model that defines the basic structure of the model.

        :param spec: more specification that completes the model
        :type spec: NNSpec
        """
        super().__init__()
        self.spec = spec


# TODO: use static_argnum instead of static_argnames
# TODO: also use action histories as biased planes
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

    def with_jit(self):
        return NNModel(
            spec=copy.deepcopy(self.spec),
            init_params_and_state=self.init_params_and_state,
            root_inference=jax.jit(self.root_inference, static_argnums=3),
            trans_inference=jax.jit(self.trans_inference, static_argnums=3),
            root_inference_unbatched=jax.jit(
                self.root_inference_unbatched, static_argnums=3
            ),
            trans_inference_unbatched=jax.jit(
                self.trans_inference_unbatched, static_argnames="is_training"
            ),
        )


def make_model(architecture_cls: Type[NNArchitecture], spec: NNSpec):
    arch = functools.partial(architecture_cls, spec)

    def multi_transform_target():
        module = arch()

        def module_walk(root_feats, trans_feats):
            root_out = module.root_inference(root_feats, is_training=True)
            trans_out = module.trans_inference(trans_feats, is_training=True)
            return (trans_out, root_out)

        return module_walk, (module.root_inference, module.trans_inference)

    transformed = hk.multi_transform_with_state(multi_transform_target)

    def init_params_and_state(rng):
        batch_dim = (1,)
        root_feats = RootFeatures(
            stacked_frames=jnp.ones(batch_dim + spec.stacked_frames_shape),
            player=jnp.array([0]),
        )
        trans_feats = TransitionFeatures(
            hidden_state=jnp.ones(
                batch_dim + spec.stacked_frames_shape[:-1] + (spec.dim_repr,)
            ),
            action=jnp.array([0]),
        )

        return transformed.init(rng, root_feats, trans_feats)

    dummy_random_key = jax.random.PRNGKey(0)

    def root_inference(params, state, feats, is_training):
        return transformed.apply[0](params, state, dummy_random_key, feats, is_training)

    def trans_inference(params, state, feats, is_training):
        return transformed.apply[1](params, state, dummy_random_key, feats, is_training)

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
    )
