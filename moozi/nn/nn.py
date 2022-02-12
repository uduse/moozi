import functools
from typing import Callable, NamedTuple, Tuple
import chex

import haiku as hk
import jax
import jax.numpy as jnp
from acme.jax.utils import add_batch_dim, squeeze_batch_dim


class NNOutput(NamedTuple):
    value: jnp.ndarray  # (batch_size, 1)
    reward: jnp.ndarray  # (batch_size, 1)
    policy_logits: jnp.ndarray  # (batch_size, num_actions)
    hidden_state: jnp.ndarray  # (batch_size, height, width, dim_repr)


class RootInferenceFeatures(NamedTuple):
    stacked_frames: jnp.ndarray  # (batch_size, num_stacked_frames, height, width, channels)
    player: jnp.ndarray  # (batch_size, 1)


class TransitionInferenceFeatures(NamedTuple):
    hidden_state: jnp.ndarray  # (batch_size, height, width, dim_repr)
    action: jnp.ndarray  # (batch_size, 1)


class NNSpec(NamedTuple):
    # define the shapes of inputs, outputs, and hidden states
    # common to all networks
    architecture: type
    stacked_frames_shape: tuple
    dim_repr: int
    dim_action: int

    # define detailed layers inside of the network
    extra: dict


# TODO: annotate this class
class NeuralNetwork(NamedTuple):
    init_network: Callable
    root_inference: Callable[..., NNOutput]
    trans_inference: Callable[..., NNOutput]
    root_inference_unbatched: Callable[..., NNOutput]
    trans_inference_unbatched: Callable[..., NNOutput]


def init_root_inference(random_key, spec, root_inference, is_training):
    dummy_batch_dim = 1
    root_inference_params, root_inference_state = root_inference.init(
        random_key,
        RootInferenceFeatures(
            stacked_frames=jnp.ones((dummy_batch_dim,) + spec.stacked_frames_shape),
            player=jnp.array(0),
        ),
        is_training,
    )

    return root_inference_params, root_inference_state


def init_trans_inference(random_key, spec, trans_inference, is_training):
    dummy_batch_dim = 1
    trans_inference_params, trans_inference_state = trans_inference.init(
        random_key,
        TransitionInferenceFeatures(
            hidden_state=jnp.ones(
                (dummy_batch_dim, *spec.stacked_frames_shape[1:3], spec.dim_repr)
            ),
            action=jnp.ones((dummy_batch_dim,)),
        ),
        is_training,
    )
    return trans_inference_params, trans_inference_state


def build_init_network_fn(spec: NNSpec, root_inference, trans_inference):
    def init_nework(random_key):
        key_1, key_2 = jax.random.split(random_key)
        root_inference_params, root_inference_state = init_root_inference(
            key_1, spec, root_inference, is_training=True
        )
        trans_inference_params, trans_inference_state = init_trans_inference(
            key_2, spec, trans_inference, is_training=True
        )

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


def build_network(spec: NNSpec):
    architecture = functools.partial(spec.architecture, spec)
    root_inference = build_root_inference(architecture)
    trans_inference = build_trans_inference(architecture)
    network_init_fn = build_init_network_fn(spec, root_inference, trans_inference)
    root_inferenc_fn_unbatched = build_unbatched_fn(root_inference.apply)
    trans_inference_fn_unbatched = build_unbatched_fn(trans_inference.apply)
    return NeuralNetwork(
        network_init_fn,
        root_inference.apply,
        trans_inference.apply,
        root_inferenc_fn_unbatched,
        trans_inference_fn_unbatched,
    )
