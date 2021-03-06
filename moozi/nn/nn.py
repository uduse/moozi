import functools
from dataclasses import dataclass
from typing import Callable, NamedTuple, Tuple, Type, Union

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from acme.jax.utils import add_batch_dim, squeeze_batch_dim

from moozi import ScalarTransform


# NOTE: NamedTuple are used for data structures that need to be passed to jax.jit functions
# TODO: could be updated using dataclass registry https://github.com/google/jax/issues/2371
class NNOutput(NamedTuple):
    """Neural network output structure.

    This is the return type of both :meth:`~moozi.nn.nn.NNArchitecture.root_inference` and
    :meth:`~moozi.nn.nn.NNArchitecture.trans_inference`.

    """

    #: (batch_size?, 1)
    value: Union[np.ndarray, jnp.ndarray]

    #: (batch_size?, 1)
    reward: Union[np.ndarray, jnp.ndarray]

    #: (batch_size?, num_actions)
    policy_logits: Union[np.ndarray, jnp.ndarray]

    #: (batch_size?, repr_rows, repr_cols, repr_channels)
    hidden_state: Union[np.ndarray, jnp.ndarray]


class RootFeatures(NamedTuple):
    """Features used in :any:`root_inference`."""

    #: (batch_size?, obs_rows, obs_cols, obs_channels)
    obs: Union[np.ndarray, jnp.ndarray]

    #: (batch_size?, 1)
    player: Union[np.ndarray, jnp.ndarray]


class TransitionFeatures(NamedTuple):
    """Features used in :any:`trans_inference`."""

    #: (batch_size?, repr_rows, repr_cols, repr_channels)
    hidden_state: Union[np.ndarray, jnp.ndarray]

    #: (batch_size?, 1)
    action: Union[np.ndarray, jnp.ndarray]


@dataclass
class NNSpec:
    """Specification of a neural network architecture.

    :any:`NNArchitecture` defines the general structure of a neural network.
    However, this is insufficient to define the exact structure of the neural network since different environments have
    differently shaped observations. Additionally, we would also like to tweak the neural network such as changing the
    number of repeating layers. As a result, we need :any:`NNSpec` to provide such additional information. Together, we
    could build a concrete neural network.

    This base class includes essential information about the shapes of inputs and outputs of the three MuZero functions.
    Other architecture-specific information could be defined in the derived classes.
    """

    obs_rows: int
    obs_cols: int
    obs_channels: int

    repr_rows: int
    repr_cols: int
    repr_channels: int

    dim_action: int

    scalar_transform: ScalarTransform


class NNArchitecture(hk.Module):
    def __init__(self, spec: NNSpec):
        """Partiallly complete neural network model that defines the basic structure of a neural network.

        Also specifies the flow of :any:`root_infercence` and :any:`trans_inference`.
        Base classes should not override these methods.

        """
        super().__init__()
        self.spec = spec

    def _repr_net(self, obs: jnp.ndarray, is_training: bool):
        # TODO: add player info
        raise NotImplementedError

    def _pred_net(self, hidden_state: jnp.ndarray, is_training: bool):
        raise NotImplementedError

    def _dyna_net(
        self, hidden_state: jnp.ndarray, action: jnp.ndarray, is_training: bool
    ):
        raise NotImplementedError

    def _proj_net(self, hidden_state: jnp.ndarray, is_training: bool):
        raise NotImplementedError

    def root_inference(self, root_feats: RootFeatures, is_training: bool):
        """Uses the representation function and the prediction function."""
        hidden_state = self._repr_net(root_feats.obs, is_training)
        value_logits, policy_logits = self._pred_net(hidden_state, is_training)
        reward_logits = jnp.zeros_like(value_logits)

        chex.assert_rank(
            [value_logits, reward_logits, policy_logits, hidden_state], [2, 2, 2, 4]
        )

        if is_training:
            value = value_logits
            reward = reward_logits
        else:
            value_probs = jax.nn.softmax(value_logits)
            reward_probs = jax.nn.softmax(reward_logits)
            value = self.spec.scalar_transform.inverse_transform(value_probs)
            reward = self.spec.scalar_transform.inverse_transform(reward_probs)

        return NNOutput(
            value=value,
            reward=reward,
            policy_logits=policy_logits,
            hidden_state=hidden_state,
        )

    def trans_inference(self, trans_feats: TransitionFeatures, is_training: bool):
        """Uses the dynamics function and the prediction function."""
        next_hidden_state, reward_logits = self._dyna_net(
            trans_feats.hidden_state,
            trans_feats.action,
            is_training,
        )
        value_logits, policy_logits = self._pred_net(next_hidden_state, is_training)
        chex.assert_rank(
            [value_logits, reward_logits, policy_logits, next_hidden_state],
            [2, 2, 2, 4],
        )

        if is_training:
            value = value_logits
            reward = reward_logits
        else:
            value_probs = jax.nn.softmax(value_logits)
            reward_probs = jax.nn.softmax(reward_logits)
            value = self.spec.scalar_transform.inverse_transform(value_probs)
            reward = self.spec.scalar_transform.inverse_transform(reward_probs)

        return NNOutput(
            value=value,
            reward=reward,
            policy_logits=policy_logits,
            hidden_state=next_hidden_state,
        )

    def projection_inference(self, hidden_state, is_training: bool):
        projection = self._proj_net(hidden_state, is_training)
        chex.assert_equal_shape([projection, hidden_state])
        return projection


# TODO: also add action histories as bias planes
@dataclass
class NNModel:
    """
    A complete neural network model defined by a collection of pure functions.

    This class should not be constructed directly. Instead, use :any:`make_model`.
    """

    #: initialize Haiku parameters and state
    init_params_and_state: Callable[[jax.random.KeyArray], Tuple[hk.Params, hk.State]]

    #: representation function + prediction function
    root_inference: Callable[
        [hk.Params, hk.State, RootFeatures, bool], Tuple[NNOutput, hk.State]
    ]

    #: dynamics function + prediction function
    trans_inference: Callable[
        [hk.Params, hk.State, TransitionFeatures, bool], Tuple[NNOutput, hk.State]
    ]

    projection_inference: Callable[
        [hk.Params, hk.State, jnp.ndarray, bool], Tuple[jnp.ndarray, hk.State]
    ]

    #: unbatched version of root_inference
    root_inference_unbatched: Callable[
        [hk.Params, hk.State, RootFeatures, bool], Tuple[NNOutput, hk.State]
    ]

    #: unbatched version of trans_inference
    trans_inference_unbatched: Callable[
        [hk.Params, hk.State, TransitionFeatures, bool], Tuple[NNOutput, hk.State]
    ]

    projection_inference_unbatched: Callable[
        [hk.Params, hk.State, jnp.ndarray, bool], Tuple[jnp.ndarray, hk.State]
    ]

    # TODO: remove hk_transformed here in the future, currently here for debugging purposes
    hk_transformed: hk.MultiTransformedWithState

    def with_jit(self) -> "NNModel":
        """Return a new model with inference functions passed through `jax.jit`."""

        # TODO: add chex assert max trace
        # NOTE: is static_argnum really should be used?
        return NNModel(
            init_params_and_state=self.init_params_and_state,
            root_inference=jax.jit(
                self.root_inference,
                static_argnums=3,
            ),
            trans_inference=jax.jit(
                self.trans_inference,
                static_argnums=3,
            ),
            projection_inference=jax.jit(
                self.projection_inference,
                static_argnums=3,
            ),
            root_inference_unbatched=jax.jit(
                self.root_inference_unbatched,
                static_argnums=3,
            ),
            trans_inference_unbatched=jax.jit(
                self.trans_inference_unbatched,
                static_argnums=3,
            ),
            projection_inference_unbatched=jax.jit(
                self.projection_inference_unbatched,
                static_argnums=3,
            ),
            hk_transformed=self.hk_transformed,
        )

    def __hash__(self) -> int:
        return 0


def make_model(architecture_cls: Type[NNArchitecture], spec: NNSpec) -> NNModel:
    """Make a concrete neural network model based on the architecture class and the specification."""

    arch = functools.partial(architecture_cls, spec)

    def multi_transform_target():
        module = arch()

        def module_walk(root_feats, trans_feats, is_training):
            root_out = module.root_inference(root_feats, is_training=is_training)
            trans_out = module.trans_inference(trans_feats, is_training=is_training)
            project_out = module._proj_net(
                trans_feats.hidden_state, is_training=is_training
            )
            return (trans_out, root_out, project_out)

        return module_walk, (
            module.root_inference,
            module.trans_inference,
            module.projection_inference,
        )

    hk_transformed = hk.multi_transform_with_state(multi_transform_target)

    def init_params_and_state(random_key):
        batch = 1
        obs_shape = (batch, spec.obs_rows, spec.obs_cols, spec.obs_channels)

        root_feats = RootFeatures(
            obs=jax.random.normal(random_key, shape=obs_shape),
            player=jnp.zeros((batch,)),
        )

        hidden_state_shape = (batch, spec.repr_rows, spec.repr_cols, spec.repr_channels)
        trans_feats = TransitionFeatures(
            hidden_state=jax.random.normal(random_key, hidden_state_shape),
            action=jnp.zeros((batch,)),
        )
        params, state = hk_transformed.init(
            random_key, root_feats, trans_feats, is_training=True
        )
        
        # initialize with a random normal distribution to workaround
        # https://github.com/deepmind/dm-haiku/issues/361
        nn_out, state = hk_transformed.apply[0](
            params, state, random_key, root_feats, is_training=True
        )
        _, state = hk_transformed.apply[1](
            params, state, random_key, trans_feats, is_training=True
        )
        return params, state

    # workaround for https://github.com/deepmind/dm-haiku/issues/325#event-6253407164
    # TODO: follow up update for the issue and make changes accordingly
    # TODO: or, alternatively, actually use random key so we could use things like dropouts
    # otherwise we can do:
    # root_inference = hk.without_apply_rng(hk_transformed).apply[0]
    # trans_inference = hk.without_apply_rng(hk_transformed).apply[1]
    dummy_random_key = jax.random.PRNGKey(0)

    def root_inference(params, state, feats, is_training):
        return hk_transformed.apply[0](
            params, state, dummy_random_key, feats, is_training
        )

    def trans_inference(params, state, feats, is_training):
        return hk_transformed.apply[1](
            params, state, dummy_random_key, feats, is_training
        )

    def projection_inference(params, state, hidden_state, is_training):
        return hk_transformed.apply[2](
            params, state, dummy_random_key, hidden_state, is_training
        )

    def root_inference_unbatched(params, state, feats, is_training):
        out, state = root_inference(params, state, add_batch_dim(feats), is_training)
        return squeeze_batch_dim(out), state

    def trans_inference_unbatched(params, state, feats, is_training):
        out, state = trans_inference(params, state, add_batch_dim(feats), is_training)
        return squeeze_batch_dim(out), state

    def projection_inference_unbatched(params, state, hidden_state, is_training):
        out, state = projection_inference(
            params, state, add_batch_dim(hidden_state), is_training
        )
        return squeeze_batch_dim(out), state

    return NNModel(
        init_params_and_state,
        root_inference,
        trans_inference,
        projection_inference,
        root_inference_unbatched,
        trans_inference_unbatched,
        projection_inference_unbatched,
        hk_transformed,
    )
