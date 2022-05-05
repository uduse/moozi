from dataclasses import dataclass
from typing import Any, Tuple

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import rlax
import tree
from jax import vmap

import moozi as mz
from moozi.logging import LogScalar, LogHistogram
from moozi.nn import RootFeatures, TransitionFeatures


class LossFn:
    def __call__(
        self,
        model: mz.nn.NNModel,
        params: hk.Params,
        state: hk.State,
        batch: mz.replay.TrainTarget,
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


@dataclass
class MuZeroLoss(LossFn):
    num_unroll_steps: int
    weight_decay: float = 1e-4

    def __call__(
        self,
        model: mz.nn.NNModel,
        params: hk.Params,
        state: hk.State,
        batch: mz.replay.TrainTarget,
    ) -> Tuple[chex.ArrayDevice, Any]:
        chex.assert_rank(
            [
                batch.stacked_frames,
                batch.action,
                batch.n_step_return,
                batch.last_reward,
                batch.action_probs,
            ],
            [4, 2, 2, 2, 3],
        )
        chex.assert_equal_shape_prefix(
            [
                batch.stacked_frames,
                batch.action,
                batch.n_step_return,
                batch.last_reward,
                batch.action_probs,
            ],
            prefix_len=1,
        )  # assert same batch dim

        init_inf_features = RootFeatures(
            obs=batch.stacked_frames,
            # TODO: actually pass player
            player=jnp.ones((batch.stacked_frames.shape[0], 1)),
        )
        is_training = True
        network_output, state = model.root_inference(
            params, state, init_inf_features, is_training
        )

        losses = {}

        losses["loss:value_0"] = vmap(mse)(
            batch.n_step_return.take(0, axis=1), network_output.value
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

            losses[f"loss:reward_{str(i + 1)}"] = (
                vmap(mse)(batch.last_reward.take(i + 1, axis=1), network_output.reward)
                * transition_loss_scale
            )
            losses[f"loss:value_{str(i + 1)}"] = (
                vmap(mse)(batch.n_step_return.take(i + 1, axis=1), network_output.value)
                * transition_loss_scale
            )
            losses[f"loss:action_probs_{str(i + 1)}"] = (
                vmap(rlax.categorical_cross_entropy)(
                    batch.action_probs.take(i + 1, axis=1), network_output.policy_logits
                )
                * transition_loss_scale
            )

        losses["loss:l2"] = jnp.reshape(
            params_l2_loss(params) * self.weight_decay, (1,)
        )
        
        importance_sampling_adjustment = 1 / batch.weight 
        loss = jnp.mean(jnp.concatenate(tree.flatten(losses)))

        losses["loss"] = loss

        step_data = {}
        for key, value in tree.map_structure(jnp.mean, losses).items():
            step_data[key] = value

        return loss, dict(state=state, step_data=step_data)
