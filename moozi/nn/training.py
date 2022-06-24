from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple, Type

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import moozi as mz
import optax
import rlax
import tree
from jax import vmap
from moozi import ScalarTransform, TrainingState, TrainTarget
from moozi.core.types import TrainingState
from moozi.nn import (
    NNArchitecture,
    NNModel,
    NNSpec,
    RootFeatures,
    TransitionFeatures,
    make_model,
)


class LossFn:
    def __call__(
        self,
        model: NNModel,
        params: hk.Params,
        state: hk.State,
        batch: TrainTarget,
    ) -> Tuple[chex.ArrayDevice, Any]:
        r"""Loss function."""
        raise NotImplementedError


def params_l2_loss(params):
    return 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))


def mse(a, b):
    return jnp.mean((a - b) ** 2)


def scale_gradient(x, scale):
    """Scales the gradient for the backward pass."""
    return x * scale + jax.lax.stop_gradient(x) * (1 - scale)


def _compute_prior_kl(
    model: NNModel, batch: TrainTarget, orig_params, new_params, state
):
    is_training = False
    orig_out, _ = model.root_inference(
        orig_params,
        state,
        RootFeatures(obs=batch.obs, player=jnp.array(0)),
        is_training,
    )
    orig_logits = orig_out.policy_logits
    new_out, _ = model.root_inference(
        new_params,
        state,
        RootFeatures(obs=batch.obs, player=jnp.array(0)),
        is_training,
    )
    new_logits = new_out.policy_logits
    prior_kl = jnp.mean(rlax.categorical_kl_divergence(orig_logits, new_logits))
    return prior_kl


# @dataclass
# class MuZeroLoss(LossFn):
#     num_unroll_steps: int
#     weight_decay: float = 1e-4

#     def __call__(
#         self,
#         model: hk.Params,
#         state: hk.State,
#         batch: TrainTarget,
#     ) -> Tuple[chex.ArrayDevice, Any]:
#         chex.assert_rank(
#             [
#                 batch.stacked_frames,
#                 batch.action,
#                 batch.n_step_return,
#                 batch.last_reward,
#                 batch.action_probs,
#             ],
#             [4, 2, 2, 2, 3],
#         )
#         chex.assert_equal_shape_prefix(
#             [
#                 batch.stacked_frames,
#                 batch.action,
#                 batch.n_step_return,
#                 batch.last_reward,
#                 batch.action_probs,
#             ],
#             prefix_len=1,
#         )  # assert same batch dim
#         batch_size = batch.stacked_frames.shape[0]

#         init_inf_features = RootFeatures(
#             obs=batch.stacked_frames,
#             # TODO: actually pass player
#             player=jnp.ones((batch.stacked_frames.shape[0], 1)),
#         )
#         is_training = True
#         network_output, state = model.root_inference(
#             params, state, init_inf_features, is_training
#         )

#         losses = {}

#         losses["loss:value_0"] = vmap(mse)(
#             batch.n_step_return.take(0, axis=1), network_output.value
#         )
#         losses["loss:action_probs_0"] = vmap(rlax.categorical_cross_entropy)(
#             batch.action_probs.take(0, axis=1), network_output.policy_logits
#         )

#         transition_loss_scale = 1 / self.num_unroll_steps
#         for i in range(self.num_unroll_steps):
#             hidden_state = scale_gradient(network_output.hidden_state, 0.5)
#             recurr_inf_features = TransitionFeatures(
#                 hidden_state=hidden_state,
#                 action=batch.action.take(0, axis=1),
#             )
#             network_output, state = model.trans_inference(
#                 params, state, recurr_inf_features, is_training
#             )

#             losses[f"loss:reward_{str(i + 1)}"] = (
#                 vmap(mse)(batch.last_reward.take(i + 1, axis=1), network_output.reward)
#                 * transition_loss_scale
#             )
#             losses[f"loss:value_{str(i + 1)}"] = (
#                 vmap(mse)(batch.n_step_return.take(i + 1, axis=1), network_output.value)
#                 * transition_loss_scale
#             )
#             losses[f"loss:action_probs_{str(i + 1)}"] = (
#                 vmap(rlax.categorical_cross_entropy)(
#                     batch.action_probs.take(i + 1, axis=1), network_output.policy_logits
#                 )
#                 * transition_loss_scale
#             )

#         # all batched losses should be the shape of (batch_size,)
#         tree.map_structure(lambda x: chex.assert_shape(x, (batch_size,)), losses)

#         losses["loss:l2"] = jnp.reshape(
#             params_l2_loss(params) * self.weight_decay, (1,)
#         )

#         # apply importance sampling adjustment
#         losses = tree.map_structure(
#             lambda x: x * batch.importance_sampling_ratio, losses
#         )

#         # sum all losses
#         loss = jnp.mean(jnp.concatenate(tree.flatten(losses)))

#         losses["loss"] = loss

#         step_data = {}
#         for key, value in tree.map_structure(jnp.mean, losses).items():
#             step_data[key] = value

#         return loss, dict(state=state, step_data=step_data)


@dataclass
class MuZeroLossWithScalarTransform(LossFn):
    num_unroll_steps: int
    scalar_transform: ScalarTransform
    weight_decay: float = 1e-4

    def __call__(
        self,
        model: NNModel,
        params: hk.Params,
        state: hk.State,
        batch: TrainTarget,
    ) -> Tuple[chex.ArrayDevice, Any]:
        chex.assert_rank(
            [
                batch.obs,
                batch.action,
                batch.n_step_return,
                batch.last_reward,
                batch.action_probs,
            ],
            [4, 2, 2, 2, 3],
        )
        chex.assert_equal_shape_prefix(
            [
                batch.obs,
                batch.action,
                batch.n_step_return,
                batch.last_reward,
                batch.action_probs,
            ],
            prefix_len=1,
        )  # assert same batch dim
        batch_size = batch.obs.shape[0]

        init_inf_features = RootFeatures(
            obs=batch.obs,
            # TODO: actually pass player
            player=jnp.ones((batch_size, 1)),
        )
        is_training = True
        network_output, state = model.root_inference(
            params, state, init_inf_features, is_training
        )

        losses = {}

        n_step_return = batch.n_step_return.take(0, axis=1)
        n_step_return_transformed = self.scalar_transform.transform(n_step_return)
        losses["loss:value_0"] = vmap(rlax.categorical_cross_entropy)(
            n_step_return_transformed, network_output.value
        )

        losses["loss:action_probs_0"] = vmap(rlax.categorical_cross_entropy)(
            batch.action_probs.take(0, axis=1), network_output.policy_logits
        )

        transition_loss_scale = 1 / self.num_unroll_steps
        for i in range(self.num_unroll_steps):
            hidden_state = scale_gradient(network_output.hidden_state, 0.5)
            recurr_inf_features = TransitionFeatures(
                hidden_state=hidden_state,
                action=batch.action.take(0, axis=1),
            )
            network_output, state = model.trans_inference(
                params, state, recurr_inf_features, is_training
            )

            reward_transformed = self.scalar_transform.transform(
                batch.last_reward.take(i + 1, axis=1)
            )
            losses[f"loss:reward_{str(i + 1)}"] = (
                vmap(rlax.categorical_cross_entropy)(
                    reward_transformed, network_output.reward
                )
                * transition_loss_scale
            )
            n_step_return_transformed = self.scalar_transform.transform(
                batch.n_step_return.take(i + 1, axis=1)
            )
            losses[f"loss:value_{str(i + 1)}"] = (
                vmap(rlax.categorical_cross_entropy)(
                    n_step_return_transformed, network_output.value
                )
                * transition_loss_scale
            )

            losses[f"loss:action_probs_{str(i + 1)}"] = (
                vmap(rlax.categorical_cross_entropy)(
                    batch.action_probs.take(i + 1, axis=1), network_output.policy_logits
                )
                * transition_loss_scale
            )

        # all batched losses should be the shape of (batch_size,)
        tree.map_structure(lambda x: chex.assert_shape(x, (batch_size,)), losses)

        losses["loss:l2"] = jnp.reshape(
            params_l2_loss(params) * self.weight_decay, (1,)
        )

        # apply importance sampling adjustment
        losses = tree.map_structure(
            lambda x: x * batch.importance_sampling_ratio, losses
        )

        # sum all losses
        loss = jnp.mean(jnp.concatenate(tree.flatten(losses)))

        losses["loss"] = loss

        step_data = {}
        for key, value in tree.map_structure(jnp.mean, losses).items():
            step_data[key] = value

        return loss, dict(state=state, step_data=step_data)


def make_sgd_step_fn(
    model: NNModel,
    loss_fn: LossFn,
    optimizer,
    target_update_period: int = 1,
    include_prior_kl: bool = True,
    include_weights: bool = False,
) -> Callable[[TrainingState, TrainTarget], Tuple[TrainingState, Dict[str, Any]]]:
    @jax.jit
    @chex.assert_max_traces(n=1)
    def sgd_step_fn(training_state: TrainingState, batch: mz.replay.TrainTarget):
        # gradient descend
        _, new_key = jax.random.split(training_state.rng_key)
        grads, extra = jax.grad(loss_fn, has_aux=True, argnums=1)(
            model, training_state.params, training_state.state, batch
        )
        updates, new_opt_state = optimizer.update(grads, training_state.opt_state)
        new_params = optax.apply_updates(training_state.params, updates)
        new_steps = training_state.steps + 1

        # TODO: put the target_update_period in the config and use it
        target_params = rlax.periodic_update(
            new_params, training_state.target_params, new_steps, target_update_period
        )

        new_training_state = TrainingState(
            params=new_params,
            target_params=target_params,
            state=extra["state"],
            opt_state=new_opt_state,
            steps=new_steps,
            rng_key=new_key,
        )

        step_data = extra["step_data"]

        if include_weights:
            for module, weight_name, weights in hk.data_structures.traverse(new_params):
                name = module + "/" + weight_name
                step_data[name] = weights

        if include_prior_kl:
            prior_kl = _compute_prior_kl(
                model, batch, training_state.params, new_params, training_state.state
            )
            step_data["prior_kl"] = prior_kl

        return new_training_state, step_data

    return sgd_step_fn


def make_training_suite(
    seed: int,
    nn_arch_cls: Type[NNArchitecture],
    nn_spec: NNSpec,
    weight_decay: float,
    lr: float,
    num_unroll_steps: int,
) -> Tuple[NNModel, TrainingState, Callable]:
    model = make_model(nn_arch_cls, nn_spec)
    random_key = jax.random.PRNGKey(seed)
    params, state = model.init_params_and_state(random_key)
    loss_fn = MuZeroLossWithScalarTransform(
        num_unroll_steps=num_unroll_steps,
        weight_decay=weight_decay,
        scalar_transform=nn_spec.scalar_transform,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1),
        optax.adam(lr, b1=0.9, b2=0.99),
    )
    training_state = TrainingState(
        params=params,
        target_params=params,
        state=state,
        opt_state=optimizer.init(params),
        steps=0,
        rng_key=jax.random.PRNGKey(seed),
    )
    sgd_step_fn = make_sgd_step_fn(model, loss_fn, optimizer)
    return model.with_jit(), training_state, sgd_step_fn
