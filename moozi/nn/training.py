from dataclasses import dataclass
from acme.utils.tree_utils import stack_sequence_fields
from typing import Any, Callable, Dict, Tuple, Type

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import moozi as mz
import optax
import rlax
import tree
from jax import vmap
from moozi import ScalarTransform, TrainingState, TrainTarget
from moozi.core.types import TrainingState, TrajectorySample
from moozi.laws import make_stacker
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

#         losses["loss/value_0"] = vmap(mse)(
#             batch.n_step_return.take(0, axis=1), network_output.value
#         )
#         losses["loss/action_probs_0"] = vmap(rlax.categorical_cross_entropy)(
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

#             losses[f"loss/reward_{str(i + 1)}"] = (
#                 vmap(mse)(batch.last_reward.take(i + 1, axis=1), network_output.reward)
#                 * transition_loss_scale
#             )
#             losses[f"loss/value_{str(i + 1)}"] = (
#                 vmap(mse)(batch.n_step_return.take(i + 1, axis=1), network_output.value)
#                 * transition_loss_scale
#             )
#             losses[f"loss/action_probs_{str(i + 1)}"] = (
#                 vmap(rlax.categorical_cross_entropy)(
#                     batch.action_probs.take(i + 1, axis=1), network_output.policy_logits
#                 )
#                 * transition_loss_scale
#             )

#         # all batched losses should be the shape of (batch_size,)
#         tree.map_structure(lambda x: chex.assert_shape(x, (batch_size,)), losses)

#         losses["loss/l2"] = jnp.reshape(
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
            player=jnp.zeros((batch_size, 1)),
        )
        is_training = True
        network_output, state = model.root_inference(
            params, state, init_inf_features, is_training
        )

        losses = {}
        info = {}

        n_step_return = batch.n_step_return.take(0, axis=1)
        n_step_return_transformed = self.scalar_transform.transform(n_step_return)
        losses["loss/value_0"] = vmap(rlax.categorical_cross_entropy)(
            labels=n_step_return_transformed,
            logits=network_output.value,
        ) * 0.25

        losses["loss/action_probs_0"] = vmap(rlax.categorical_cross_entropy)(
            labels=batch.action_probs.take(0, axis=1),
            logits=network_output.policy_logits,
        )

        transition_loss_scale = 1 / self.num_unroll_steps
        for i in range(self.num_unroll_steps):
            hidden_state = scale_gradient(network_output.hidden_state, 0.5)
            if "hidden_state" not in info:
                info["hidden_state"] = hidden_state
            trans_feats = TransitionFeatures(
                hidden_state=hidden_state,
                action=batch.action.take(i, axis=1),
            )
            network_output, state = model.trans_inference(
                params, state, trans_feats, is_training
            )

            # reward loss
            reward_transformed = self.scalar_transform.transform(
                batch.last_reward.take(i + 1, axis=1)
            )
            chex.assert_shape(
                reward_transformed,
                (batch_size, self.scalar_transform.dim),
            )
            losses[f"loss/reward_{str(i + 1)}"] = (
                vmap(rlax.categorical_cross_entropy)(
                    labels=reward_transformed,
                    logits=network_output.reward,
                )
                * transition_loss_scale
            )

            # value loss
            n_step_return_transformed = self.scalar_transform.transform(
                batch.n_step_return.take(i + 1, axis=1)
            )
            chex.assert_shape(
                n_step_return_transformed,
                (batch_size, self.scalar_transform.dim),
            )
            losses[f"loss/value_{str(i + 1)}"] = (
                vmap(rlax.categorical_cross_entropy)(
                    labels=n_step_return_transformed,
                    logits=network_output.value,
                )
                * transition_loss_scale * 0.25
            )

            # action probs loss
            losses[f"loss/action_probs_{str(i + 1)}"] = (
                vmap(rlax.categorical_cross_entropy)(
                    batch.action_probs.take(i + 1, axis=1),
                    network_output.policy_logits,
                )
                * transition_loss_scale
            )

        # all batched losses should be the shape of (batch_size,)
        tree.map_structure(lambda x: chex.assert_shape(x, (batch_size,)), losses)

        losses["loss/l2"] = jnp.reshape(
            params_l2_loss(params) * self.weight_decay, (1,)
        )

        # apply importance sampling adjustment
        losses = tree.map_structure(
            lambda x: x * batch.importance_sampling_ratio, losses
        )
        info["importance_sampling_ratio"] = batch.importance_sampling_ratio

        # sum all losses
        loss = jnp.mean(jnp.concatenate(tree.flatten(losses)))

        losses["loss"] = loss

        step_data = {}
        for key, value in tree.map_structure(jnp.mean, losses).items():
            step_data[key] = value
        for key in info:
            step_data[f"info/{key}"] = info[key]

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
    # TODO: set target update period
    sgd_step_fn = make_sgd_step_fn(model, loss_fn, optimizer, target_update_period=20)
    return model.with_jit(), training_state, sgd_step_fn


def make_target_from_traj(
    sample: TrajectorySample,
    start_idx,
    discount,
    num_unroll_steps,
    num_td_steps,
    num_stacked_frames,
) -> TrainTarget:
    # assert not batched
    assert len(sample.last_reward.shape) == 1

    last_step_idx = sample.is_last.argmax()

    _, num_rows, num_cols, num_channels = sample.frame.shape
    dim_action = sample.action_probs.shape[-1]
    stacker = make_stacker(
        num_rows=num_rows,
        num_cols=num_cols,
        num_channels=num_channels,
        num_stacked_frames=num_stacked_frames,
        dim_action=dim_action,
    )

    frame_idx_lower = max(start_idx - num_stacked_frames + 1, 0)
    frame_idx_upper = start_idx + 1

    tape = stacker.malloc()
    for i in range(frame_idx_lower, frame_idx_upper):
        tape["frame"] = sample.frame[i]
        tape["action"] = sample.action[i]
        tape = stacker.apply(tape)
    stacked_frames = tape["stacked_frames"]
    stacked_actions = tape["stacked_actions"]
    obs = np.concatenate([stacked_frames, stacked_actions], axis=-1)

    action = _get_action(sample, start_idx, num_unroll_steps)

    unrolled_data = []
    for curr_idx in range(start_idx, start_idx + num_unroll_steps + 1):
        n_step_return = _get_n_step_return(
            sample, curr_idx, last_step_idx, num_td_steps, discount
        )
        last_reward = _get_last_reward(sample, start_idx, curr_idx, last_step_idx)
        action_probs = _get_action_probs(sample, curr_idx, last_step_idx)
        root_value = _get_root_value(sample, curr_idx, last_step_idx)
        unrolled_data.append((n_step_return, last_reward, action_probs, root_value))

    unrolled_data_stacked = stack_sequence_fields(unrolled_data)

    return TrainTarget(
        obs=obs,
        action=action,
        n_step_return=unrolled_data_stacked[0],
        last_reward=unrolled_data_stacked[1],
        action_probs=unrolled_data_stacked[2],
        root_value=unrolled_data_stacked[3],
        importance_sampling_ratio=np.ones((1,)),
    )


def _get_action(sample: TrajectorySample, start_idx, num_unroll_steps):
    action = sample.action[start_idx : start_idx + num_unroll_steps]
    num_actions_to_pad = num_unroll_steps - action.size
    if num_actions_to_pad > 0:
        action = np.concatenate((action, np.full(num_actions_to_pad, -1)))
    return action


def _get_n_step_return(
    sample: TrajectorySample,
    curr_idx,
    last_step_idx,
    num_td_steps,
    discount,
):
    """The observed N-step return with bootstrapping."""
    if curr_idx >= last_step_idx:
        return 0

    bootstrap_idx = curr_idx + num_td_steps

    accumulated_reward = _get_accumulated_reward(
        sample, curr_idx, discount, bootstrap_idx
    )
    bootstrap_value = _get_bootstrap_value(
        sample, last_step_idx, num_td_steps, discount, bootstrap_idx
    )

    return accumulated_reward + bootstrap_value


def _get_bootstrap_value(
    sample: TrajectorySample,
    last_step_idx,
    num_td_steps,
    discount,
    bootstrap_idx,
) -> float:
    if bootstrap_idx <= last_step_idx:
        value = sample.root_value[bootstrap_idx] * (discount ** num_td_steps)
        if sample.to_play[bootstrap_idx] != mz.BASE_PLAYER:
            return -value
        else:
            return value
    else:
        return 0.0


def _get_accumulated_reward(
    sample: TrajectorySample, curr_idx, discount, bootstrap_idx
) -> float:
    reward_sum = 0.0
    last_rewards = sample.last_reward[curr_idx + 1 : bootstrap_idx + 1]
    players_of_last_rewards = sample.to_play[curr_idx:bootstrap_idx]
    for i, (last_rewrad, player) in enumerate(
        zip(last_rewards, players_of_last_rewards)
    ):
        discounted_reward = last_rewrad * (discount ** i)
        reward_sum += discounted_reward
    return reward_sum


def _get_last_reward(sample: TrajectorySample, start_idx, curr_idx, last_step_idx):
    if curr_idx == start_idx:
        return 0
    elif curr_idx <= last_step_idx:
        return sample.last_reward[curr_idx]
    else:
        return 0


def _get_action_probs(sample: TrajectorySample, curr_idx, last_step_idx):
    if curr_idx <= last_step_idx:
        return sample.action_probs[curr_idx]
    else:
        return np.zeros_like(sample.action_probs[0])


def _get_root_value(sample: TrajectorySample, curr_idx, last_step_idx):
    if curr_idx <= last_step_idx:
        return sample.root_value[curr_idx]
    else:
        return 0.0
