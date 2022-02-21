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
    value: jnp.ndarray  # (batch_size, 1)
    reward: jnp.ndarray  # (batch_size, 1)
    policy_logits: jnp.ndarray  # (batch_size, num_actions)
    hidden_state: jnp.ndarray  # (batch_size, height, width, dim_repr)


class RootFeatures(NamedTuple):
    stacked_frames: jnp.ndarray  # (batch_size, height, width, channels)
    player: jnp.ndarray  # (batch_size, 1)


class TransitionFeatures(NamedTuple):
    hidden_state: jnp.ndarray  # (batch_size, height, width, dim_repr)
    action: jnp.ndarray  # (batch_size, 1)


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


@dataclass
class NNModel:
    """
    Complete neural network model that's ready for initiailization and inference.

    Note that inference functions could be jitted, but need to pass with 
    `jax.jit(..., static_argnames="is_training")` to make `is_training` static.
    """

    spec: NNSpec
    init_model: Callable
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


def init_root_inference(random_key, spec, root_inference, is_training):
    dummy_batch_dim = 1
    root_inference_params, root_inference_state = root_inference.init(
        random_key,
        RootFeatures(
            stacked_frames=jnp.ones((dummy_batch_dim,) + spec.stacked_frames_shape),
            player=jnp.array(0),
        ),
        is_training,
    )

    return root_inference_params, root_inference_state


def init_trans_inference(random_key, spec: NNSpec, trans_inference, is_training):
    dummy_batch_dim = 1
    height, width, _ = spec.stacked_frames_shape
    trans_inference_params, trans_inference_state = trans_inference.init(
        random_key,
        TransitionFeatures(
            hidden_state=jnp.ones((dummy_batch_dim, height, width, spec.dim_repr)),
            action=jnp.ones((dummy_batch_dim,)),
        ),
        is_training,
    )
    return trans_inference_params, trans_inference_state


def validate_shapes(x: Union[hk.Params, hk.State], y: Union[hk.Params, hk.State]):
    x_paths = set([k for k, _ in tree.flatten_with_path(x)])
    y_paths = set([k for k, _ in tree.flatten_with_path(y)])
    shared_paths = x_paths & y_paths
    for path in shared_paths:
        x_val, y_val = x, y
        for key in path:
            x_val = x_val[key]
            y_val = y_val[key]
        assert x_val.shape == y_val.shape, (x_val.shape, y_val.shape)


def build_init_network_fn(spec: NNSpec, root_inference, trans_inference):
    def init_nework(random_key):
        key_1, key_2 = jax.random.split(random_key)
        root_inference_params, root_inference_state = init_root_inference(
            key_1, spec, root_inference, is_training=True
        )
        trans_inference_params, trans_inference_state = init_trans_inference(
            key_2, spec, trans_inference, is_training=True
        )

        validate_shapes(root_inference_params, trans_inference_params)
        validate_shapes(root_inference_state, trans_inference_state)
        merged_params = hk.data_structures.merge(
            root_inference_params,
            trans_inference_params,
        )
        merged_state = hk.data_structures.merge(
            root_inference_state,
            trans_inference_state,
        )
        return merged_params, merged_state

    return init_nework


def build_root_inference(arch):
    return hk.without_apply_rng(
        hk.transform_with_state(
            lambda root_inf_feats, is_training: arch().initial_inference(
                root_inf_feats, is_training
            )
        )
    )


def build_trans_inference(arch):
    return hk.without_apply_rng(
        hk.transform_with_state(
            lambda trans_inf_feats, is_training: arch().recurrent_inference(
                trans_inf_feats, is_training
            )
        )
    )


def build_unbatched_fn(fn):
    def _unbatched_wrapper(params, state, feats, is_training):
        nn_out, new_state = fn(params, state, add_batch_dim(feats), is_training)
        return squeeze_batch_dim(nn_out), new_state

    return _unbatched_wrapper


def build_model(architecture_cls: Type[NNArchitecture], spec: NNSpec):
    arch = functools.partial(architecture_cls, spec)
    root_inference = build_root_inference(arch)
    trans_inference = build_trans_inference(arch)
    network_init_fn = build_init_network_fn(spec, root_inference, trans_inference)
    root_inferenc_fn_unbatched = build_unbatched_fn(root_inference.apply)
    trans_inference_fn_unbatched = build_unbatched_fn(trans_inference.apply)
    return NNModel(
        spec,
        network_init_fn,
        root_inference.apply,
        trans_inference.apply,
        root_inferenc_fn_unbatched,
        trans_inference_fn_unbatched,
    )
