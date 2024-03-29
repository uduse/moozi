from dataclasses import dataclass
from flax import struct
from acme.utils.tree_utils import stack_sequence_fields, unstack_sequence_fields
from typing import Any, Callable, Dict, List, Tuple, Type

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
from moozi.core import scalar_transform
from moozi.core.types import BASE_PLAYER, TrainingState, TrajectorySample
from moozi.core.utils import make_one_hot_planes, make_frame_planes
from moozi.laws import concat_stacked_to_obs, make_stacker
from moozi.nn import (
    NNArchitecture,
    NNModel,
    NNSpec,
    RootFeatures,
    TransitionFeatures,
    make_model,
)


def params_l2_loss(params):
    return 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))


def mse(a, b):
    return jnp.mean((a - b) ** 2)


def scale_gradient(x, scale):
    """Scales the gradient for the backward pass."""
    return x * scale + jax.lax.stop_gradient(x) * (1 - scale)


def _compute_prior_kl(
    model: NNModel,
    orig_params,
    orig_state,
    orig_feats,
    new_params,
    new_state,
    new_feats,
):
    is_training = False

    orig_out, _ = model.root_inference(
        orig_params,
        orig_state,
        orig_feats,
        is_training,
    )
    orig_logits = orig_out.policy_logits

    new_out, _ = model.root_inference(
        new_params,
        new_state,
        new_feats,
        is_training,
    )
    new_logits = new_out.policy_logits

    prior_kl = jnp.mean(rlax.categorical_kl_divergence(orig_logits, new_logits))
    return prior_kl


def _make_obs_from_train_target(
    batch: TrainTarget,
    step: int,
    history_length: int,
    num_unroll_steps: int,
    dim_action: int,
):
    (
        batch_size,
        num_frames,
        num_rows,
        num_cols,
        num_channels,
    ) = batch.frame.shape

    assert 0 <= step <= num_unroll_steps
    assert num_frames == (history_length + 1)

    history_frames = batch.frame[:, step : step + history_length, ...]
    stacked_frames = vmap(make_frame_planes)(history_frames)
    history_actions = batch.action[:, step : step + history_length, ...]
    stacked_actions = vmap(make_one_hot_planes, in_axes=[0, None, None, None])(
        history_actions, num_rows, num_cols, dim_action
    )
    obs = jnp.concatenate([stacked_frames, stacked_actions], axis=-1)
    chex.assert_shape(
        obs,
        (
            batch_size,
            num_rows,
            num_cols,
            history_length * (num_channels + dim_action),
        ),
    )
    return obs


def _slice_frames(
    batch: TrainTarget,
    step: int,
    history_length: int,
):
    return batch.frame[:, step : step + history_length, ...]


def _slice_actions(
    batch: TrainTarget,
    step: int,
    history_length: int,
):
    return batch.action[:, step : step + history_length, ...]


class LossFn(struct.PyTreeNode):
    model: NNModel
    num_unroll_steps: int
    weight_decay: float = 1e-4
    consistency_loss_coef: float = 1.0
    value_loss_coef: float = 1.0
    use_importance_sampling: bool = True

    def __call__(
        self,
        params: hk.Params,
        state: hk.State,
        batch: TrainTarget,
    ) -> Tuple[chex.ArrayDevice, Any]:
        self._check_shapes(batch)
        batch_size = batch.frame.shape[0]
        scalar_transform = self.model.spec.scalar_transform

        history_length = self.model.spec.history_length
        init_inf_features = RootFeatures(
            frames=_slice_frames(batch, 0, history_length),
            actions=_slice_actions(batch, 0, history_length),
            to_play=batch.to_play,
        )
        is_training = True
        network_output, state = self.model.root_inference(
            params, state, init_inf_features, is_training
        )

        losses = {}
        info = {}

        n_step_return = batch.n_step_return.take(0, axis=1)
        n_step_return_transformed = scalar_transform.transform(n_step_return)
        losses["loss/value_0"] = (
            optax.softmax_cross_entropy(
                labels=n_step_return_transformed,
                logits=network_output.value,
            )
            * self.value_loss_coef
        )

        losses["loss/action_probs_0"] = optax.softmax_cross_entropy(
            labels=batch.action_probs.take(0, axis=1),
            logits=network_output.policy_logits,
        )

        transition_loss_scale = 1 / self.num_unroll_steps
        for i in range(self.num_unroll_steps):
            hidden_state = scale_gradient(network_output.hidden_state, 0.5)

            if i == 0 or i == 1:
                info[f"hidden_state_{i}"] = hidden_state
            trans_feats = TransitionFeatures(
                hidden_state=hidden_state,
                action=batch.action.take(i + history_length, axis=1),
            )
            network_output, state = self.model.trans_inference(
                params, state, trans_feats, is_training
            )

            # reward loss
            reward_transformed = scalar_transform.transform(
                batch.last_reward.take(i + 1, axis=1)
            )
            chex.assert_shape(
                reward_transformed,
                (batch_size, scalar_transform.dim),
            )
            losses[f"loss/reward_{str(i + 1)}"] = (
                optax.softmax_cross_entropy(
                    labels=reward_transformed,
                    logits=network_output.reward,
                )
                * transition_loss_scale
            )

            # value loss
            n_step_return_transformed = scalar_transform.transform(
                batch.n_step_return.take(i + 1, axis=1)
            )
            chex.assert_shape(
                n_step_return_transformed,
                (batch_size, scalar_transform.dim),
            )
            losses[f"loss/value_{str(i + 1)}"] = (
                optax.softmax_cross_entropy(
                    labels=n_step_return_transformed,
                    logits=network_output.value,
                )
                * transition_loss_scale
                * self.value_loss_coef
            )

            # action probs loss
            losses[f"loss/action_probs_{str(i + 1)}"] = (
                optax.softmax_cross_entropy(
                    labels=batch.action_probs.take(i + 1, axis=1),
                    logits=network_output.policy_logits,
                )
                * transition_loss_scale
            )

            if self.consistency_loss_coef > 0.0 and i == 0:
                # flip player
                # TODO: only use in two players games
                next_player = jnp.logical_not(batch.to_play).astype(jnp.int32)
                next_feats = RootFeatures(
                    frames=_slice_frames(batch, 1, history_length),
                    actions=_slice_actions(batch, 1, history_length),
                    to_play=next_player,
                )

                next_network_output, state = self.model.root_inference(
                    params, state, next_feats, is_training
                )
                next_hidden_state = jax.lax.stop_gradient(
                    next_network_output.hidden_state
                )

                curr_projection, state = self.model.projection_inference(
                    params, state, network_output.hidden_state, is_training
                )
                # next_projection, state = model.projection_inference(
                #     params, state, next_hidden_state, is_training
                # )
                consistency_loss = (
                    optax.cosine_distance(
                        curr_projection.reshape((batch_size, -1)),
                        next_hidden_state.reshape((batch_size, -1)),
                    )
                    * transition_loss_scale
                    * self.consistency_loss_coef
                )

                losses[f"loss/consistency_{str(i + 1)}"] = consistency_loss

        # all batched losses should be the shape of (batch_size,)
        tree.map_structure(lambda x: chex.assert_shape(x, (batch_size,)), losses)

        # apply importance sampling adjustment
        if self.use_importance_sampling:
            losses = tree.map_structure(
                lambda x: x * batch.importance_sampling_ratio, losses
            )
            info["is_ratio/hist"] = batch.importance_sampling_ratio
            info["is_ratio/mean"] = jnp.mean(batch.importance_sampling_ratio)
            info["is_ratio/max"] = jnp.max(batch.importance_sampling_ratio)
            info["is_ratio/min"] = jnp.min(batch.importance_sampling_ratio)

        losses = tree.map_structure(lambda x: jnp.mean(x), losses)
        losses["loss/l2"] = params_l2_loss(params) * self.weight_decay
        loss = jnp.sum(jnp.stack(tree.flatten(losses)))
        losses["loss"] = loss

        step_data = {}
        for key, value in tree.map_structure(jnp.mean, losses).items():
            step_data[key] = value
        for key in info:
            step_data[f"info/{key}"] = info[key]

        return loss, dict(state=state, step_data=step_data)

    def _check_shapes(self, batch):

        chex.assert_rank(
            [
                batch.frame,
                batch.action,
                batch.n_step_return,
                batch.root_value,
                batch.last_reward,
                batch.action_probs,
            ],
            [5, 2, 2, 2, 2, 3],
        )

        # assert same batch dim
        chex.assert_equal_shape_prefix(
            [
                batch.frame,
                batch.action,
                batch.n_step_return,
                batch.root_value,
                batch.last_reward,
                batch.action_probs,
            ],
            prefix_len=1,
        )

        batch_size = batch.frame.shape[0]

        B = batch_size
        L = self.history_length
        K = self.num_unroll_steps
        H = self.model.spec.frame_rows
        W = self.model.spec.frame_cols
        C = self.model.spec.frame_channels
        Z = self.model.spec.scalar_transform.dim
        chex.assert_shape(batch.frame, (B, H, W, L + 1, C))
        chex.assert_shape(batch.action, (B, L + K))
        chex.assert_shape(batch.n_step_return, (B, K + 1))


class SGD(struct.PyTreeNode):
    loss_fn: LossFn
    optimizer: optax.GradientTransformation
    num_unroll_steps: int
    target_update_period: int = 1
    log_prior_kl: bool = True
    log_weights: bool = False

    def step(
        self,
        training_state: TrainingState,
        batch: TrainTarget,
    ):
        model = self.loss_fn.model
        orig_params = training_state.params
        orig_state = training_state.state

        # gradient descend
        _, new_key = jax.random.split(training_state.rng_key)
        grads, extra = jax.grad(self.loss_fn, has_aux=True, argnums=1)(
            model, training_state.params, training_state.state, batch
        )
        new_state = extra["state"]
        updates, new_opt_state = self.optimizer.update(grads, training_state.opt_state)
        new_params = optax.apply_updates(training_state.params, updates)
        new_steps = training_state.steps + 1

        target_params = rlax.periodic_update(
            new_params,
            training_state.target_params,
            new_steps,
            self.target_update_period,
        )

        target_state = rlax.periodic_update(
            new_state,
            training_state.state,
            new_steps,
            self.target_update_period,
        )

        new_training_state = TrainingState(
            params=new_params,
            target_params=target_params,
            state=new_state,
            target_state=target_state,
            opt_state=new_opt_state,
            steps=new_steps,
            rng_key=new_key,
        )

        step_data = extra["step_data"]

        if self.log_weights:
            for module, weight_name, weights in hk.data_structures.traverse(new_params):
                name = module + "/" + weight_name
                step_data[name] = weights

        history_length = model.spec.history_length
        if self.log_prior_kl:
            orig_feats = RootFeatures(
                frames=_slice_frames(batch, 0, history_length),
                actions=_slice_actions(batch, 0, history_length),
                to_play=batch.to_play,
            )

            next_player = jnp.logical_not(batch.to_play).astype(jnp.int32)
            new_feats = RootFeatures(
                frames=_slice_frames(batch, 1, history_length),
                actions=_slice_actions(batch, 1, history_length),
                to_play=next_player,
            )

            prior_kl = _compute_prior_kl(
                model=model,
                orig_params=orig_params,
                orig_state=orig_state,
                orig_feats=orig_feats,
                new_state=new_state,
                new_params=new_params,
                new_feats=new_feats,
            )
            step_data["info/prior_kl"] = prior_kl

        step_data["info/update_size"] = sum(
            jnp.sum(jnp.abs(p)) for p in jax.tree_leaves(updates)
        )

        return new_training_state, step_data


# def make_sgd_step_fn(
#     model: NNModel,
#     loss_fn: LossFn,
#     optimizer,
#     num_unroll_steps: int,
#     history_length: int,
#     dim_action: int,
#     target_update_period: int = 1,
#     include_prior_kl: bool = True,
#     include_weights: bool = False,
# ) -> Callable[[TrainingState, TrainTarget], Tuple[TrainingState, Dict[str, Any]]]:
#     @jax.jit
#     @chex.assert_max_traces(n=1)
#     def sgd_step_fn(training_state: TrainingState, batch: mz.replay.TrainTarget):
#         orig_params = training_state.params
#         orig_state = training_state.state

#         # gradient descend
#         _, new_key = jax.random.split(training_state.rng_key)
#         grads, extra = jax.grad(loss_fn, has_aux=True, argnums=1)(
#             model, training_state.params, training_state.state, batch
#         )
#         new_state = extra["state"]
#         updates, new_opt_state = optimizer.update(grads, training_state.opt_state)
#         new_params = optax.apply_updates(training_state.params, updates)
#         new_steps = training_state.steps + 1

#         target_params = rlax.periodic_update(
#             new_params,
#             training_state.target_params,
#             new_steps,
#             target_update_period,
#         )

#         target_state = rlax.periodic_update(
#             new_state,
#             training_state.state,
#             new_steps,
#             target_update_period,
#         )

#         new_training_state = TrainingState(
#             params=new_params,
#             target_params=target_params,
#             state=new_state,
#             target_state=target_state,
#             opt_state=new_opt_state,
#             steps=new_steps,
#             rng_key=new_key,
#         )

#         step_data = extra["step_data"]

#         if include_weights:
#             for module, weight_name, weights in hk.data_structures.traverse(new_params):
#                 name = module + "/" + weight_name
#                 step_data[name] = weights

#         if include_prior_kl:
#             orig_feats = RootFeatures(
#                 frames=_slice_frames(batch, 0, history_length),
#                 actions=_slice_actions(batch, 0, history_length),
#                 to_play=batch.to_play,
#             )

#             next_player = jnp.logical_not(batch.to_play).astype(jnp.int32)
#             new_feats = RootFeatures(
#                 frames=_slice_frames(batch, 1, history_length),
#                 actions=_slice_actions(batch, 1, history_length),
#                 to_play=next_player,
#             )

#             prior_kl = _compute_prior_kl(
#                 model=model,
#                 orig_params=orig_params,
#                 orig_state=orig_state,
#                 orig_feats=orig_feats,
#                 new_state=new_state,
#                 new_params=new_params,
#                 new_feats=new_feats,
#             )
#             step_data["info/prior_kl"] = prior_kl

#         step_data["info/update_size"] = sum(
#             jnp.sum(jnp.abs(p)) for p in jax.tree_leaves(updates)
#         )
#         return new_training_state, step_data
#     return sgd_step_fn


# def make_training_suite(
#     seed: int,
#     model: NNModel,
#     weight_decay: float,
#     lr: float,
#     num_unroll_steps: int,
#     history_length: int,
#     target_update_period: int = 1,
#     consistency_loss_coef: float = 1.0,
#     value_loss_coef: float = 1.0,
# ) -> Tuple[NNModel, TrainingState, Callable]:
#     random_key = jax.random.PRNGKey(seed)
#     random_key, new_key = jax.random.split(random_key)
#     params, state = model.init_params_and_state(new_key)
#     loss_fn = LossFn(
#         history_length=history_length,
#         num_unroll_steps=num_unroll_steps,
#         scalar_transform=model.spec.scalar_transform,
#         weight_decay=weight_decay,
#         dim_action=model.spec.dim_action,
#         consistency_loss_coef=consistency_loss_coef,
#         value_loss_coef=value_loss_coef,
#     )
#     optimizer = optax.chain(
#         optax.clip_by_global_norm(5),
#         optax.adam(learning_rate=lr),
#     )
#     training_state = TrainingState(
#         params=params,
#         target_params=params,
#         state=state,
#         target_state=state,
#         opt_state=optimizer.init(params),
#         steps=0,
#         rng_key=random_key,
#     )
#     sgd_step_fn = make_sgd_step_fn(
#         model=model,
#         loss_fn=loss_fn,
#         optimizer=optimizer,
#         history_length=history_length,
#         num_unroll_steps=num_unroll_steps,
#         dim_action=model.spec.dim_action,
#         target_update_period=target_update_period,
#     )
#     return model.with_jit(), training_state, sgd_step_fn


# TODO: use this trainer instead
class Trainer(struct.PyTreeNode):
    optimizer: optax.GradientTransformation
    loss_fn: LossFn
    sgd: SGD

    @staticmethod
    def new(
        model: NNModel,
        num_unroll_steps: int,
        target_update_period: int = 1,
        # loss fn
        weight_decay: float = 1e-4,
        consistency_loss_coef: float = 1.0,
        value_loss_coef: float = 1.0,
        # optimizer
        lr: float = 1e-3,
        clip_gradient: float = 5.0,
        # logging
        log_prior_kl: bool = True,
        log_weights: bool = False,
    ):
        loss_fn = LossFn(
            model=model,
            num_unroll_steps=num_unroll_steps,
            weight_decay=weight_decay,
            consistency_loss_coef=consistency_loss_coef,
            value_loss_coef=value_loss_coef,
            # use_importance_sampling: bool = True
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(clip_gradient),
            optax.adam(learning_rate=lr),
        )
        sgd = SGD(
            loss_fn=loss_fn,
            optimizer=optimizer,
            num_unroll_steps=num_unroll_steps,
            target_update_period=target_update_period,
            log_prior_kl=log_prior_kl,
            log_weights=log_weights,
        )
        return Trainer(
            optimizer=optimizer,
            loss_fn=loss_fn,
            sgd=sgd,
        )

    def init(self, random_key: chex.PRNGKey) -> TrainingState:
        params, state = self.loss_fn.model.init_params_and_state(random_key)
        return TrainingState(
            params=params,
            target_params=params,
            state=state,
            target_state=state,
            opt_state=self.optimizer.init(params),
            steps=0,
            rng_key=random_key,
        )


def make_target_from_traj(
    traj: TrajectorySample,
    start_idx,
    discount,
    num_unroll_steps,
    num_td_steps,
    history_length,
) -> TrainTarget:
    # TODO: make a JIT-able version of this

    # assert not batched
    assert len(traj.last_reward.shape) == 1
    assert sum(traj.is_last) == 1
    assert traj.is_last[-1] == True

    frame = _make_frame(traj, start_idx, history_length)
    action = _make_action(traj, start_idx, history_length, num_unroll_steps)
    to_play = _make_to_play(traj, start_idx)

    unrolled_data = []
    for curr_idx in range(start_idx, start_idx + num_unroll_steps + 1):
        n_step_return = _make_n_step_return(traj, curr_idx, num_td_steps, discount)
        last_reward = _make_last_reward(traj, curr_idx)
        action_probs = _make_action_probs(traj, curr_idx)
        root_value = _make_root_value(traj, curr_idx)
        unrolled_data.append((n_step_return, last_reward, action_probs, root_value))

    unrolled_data_stacked = stack_sequence_fields(unrolled_data)

    return TrainTarget(
        frame=frame,
        action=action,
        n_step_return=unrolled_data_stacked[0],
        last_reward=unrolled_data_stacked[1],
        action_probs=unrolled_data_stacked[2],
        root_value=unrolled_data_stacked[3],
        to_play=to_play,
        importance_sampling_ratio=np.array(1),
    )


def _make_to_play(traj: TrajectorySample, start_idx) -> np.ndarray:
    last_step_idx = traj.frame.shape[0] - 1
    if start_idx >= last_step_idx:
        return np.array(BASE_PLAYER)
    else:
        return traj.to_play[start_idx]


def _make_frame(traj: TrajectorySample, start_idx, history_length):
    # history_length frames + one extra frame for consistency loss
    frames = []
    first_frame_idx = start_idx - history_length + 1
    last_frame_idx = start_idx + 1
    for i in range(first_frame_idx, last_frame_idx + 1):
        if i < 0 or i >= traj.frame.shape[0]:
            frames.append(np.zeros_like(traj.frame[0]))
        elif i < traj.frame.shape[0]:
            frames.append(traj.frame[i])
    return stack_sequence_fields(frames)


def _make_action(traj: TrajectorySample, start_idx, history_length, num_unroll_steps):
    first_frame_idx = start_idx - history_length
    last_frame_idx = start_idx + num_unroll_steps - 1
    last_step_idx = traj.action.shape[0] - 1
    actions = []
    for i in range(first_frame_idx, last_frame_idx + 1):
        if i < 0 or i >= last_step_idx:
            actions.append(np.zeros_like(traj.action[0]))
        elif i < traj.action.shape[0]:
            actions.append(traj.action[i])
    return stack_sequence_fields(actions)


# def _make_to_play(traj: TrajectorySample, start_idx):
#     return traj.to_play[start_idx]


def _make_n_step_return(
    traj: TrajectorySample,
    curr_idx,
    num_td_steps,
    discount,
):
    """The observed N-step return with bootstrapping."""
    last_step_idx = traj.frame.shape[0] - 1
    if curr_idx >= last_step_idx:
        return 0

    accumulated_reward = _make_accumulated_reward(
        traj,
        curr_idx,
        num_td_steps=num_td_steps,
        discount=discount,
    )
    bootstrap_value = _make_bootstrap_value(
        traj,
        curr_idx,
        num_td_steps=num_td_steps,
        discount=discount,
    )

    return accumulated_reward + bootstrap_value


def _make_accumulated_reward(
    traj: TrajectorySample,
    curr_idx,
    num_td_steps,
    discount,
) -> float:
    bootstrap_idx = curr_idx + num_td_steps
    reward_sum = 0.0
    last_rewards = traj.last_reward[curr_idx + 1 : bootstrap_idx + 1]
    for i, last_reward in enumerate(last_rewards):
        discounted_reward = last_reward * (discount ** i)
        reward_sum += discounted_reward
    return reward_sum


def _make_bootstrap_value(
    traj: TrajectorySample,
    curr_idx,
    num_td_steps,
    discount,
) -> float:
    last_step_idx = traj.frame.shape[0] - 1
    bootstrap_idx = curr_idx + num_td_steps
    if bootstrap_idx <= last_step_idx:
        # NOTE: multi player flip value
        return traj.root_value[bootstrap_idx] * (discount ** num_td_steps)
    else:
        return 0.0


def _make_last_reward(traj: TrajectorySample, curr_idx):
    last_step_idx = traj.frame.shape[0] - 1
    if curr_idx <= last_step_idx:
        return traj.last_reward[curr_idx]
    else:
        return 0


def _make_action_probs(traj: TrajectorySample, curr_idx):
    last_step_idx = traj.frame.shape[0] - 1
    if curr_idx < last_step_idx:
        return traj.action_probs[curr_idx]
    else:
        # the terminal action is 0
        vec = np.zeros_like(traj.action_probs[0])
        vec[0] = 1.0
        return vec


def _make_root_value(traj: TrajectorySample, curr_idx):
    last_step_idx = traj.frame.shape[0] - 1
    if curr_idx <= last_step_idx:
        return traj.root_value[curr_idx]
    else:
        return 0.0
