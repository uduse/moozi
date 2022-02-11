from typing import Any, Tuple

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import rlax
import tree
from jax import vmap

import moozi as mz
from moozi.logging import JAXBoardStepData
from moozi.nn import RootInferenceFeatures, TransitionInferenceFeatures


class LossFn(object):
    def __call__(
        self,
        network: mz.nn.NeuralNetwork,
        params: chex.ArrayTree,
        batch: mz.replay.TrainTarget,
    ) -> Tuple[chex.ArrayDevice, Any]:
        r"""Loss function."""
        raise NotImplementedError


def params_l2_loss(params):
    return 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))


def mse(a, b):
    return jnp.mean((a - b) ** 2)


class MuZeroLoss(LossFn):
    def __init__(self, num_unroll_steps, weight_decay=1e-4):
        self._num_unroll_steps = num_unroll_steps
        self._weight_decay = weight_decay

    def __call__(
        self,
        network: mz.nn.NeuralNetwork,
        params: hk.Params,
        state: hk.State,
        batch: mz.replay.TrainTarget,
    ) -> Tuple[chex.ArrayDevice, Any]:

        chex.assert_rank(
            [
                batch.stacked_frames,
                batch.action,
                batch.value,
                batch.last_reward,
                batch.action_probs,
            ],
            [3, 2, 2, 2, 3],
        )

        init_inf_features = RootInferenceFeatures(
            stacked_frames=batch.stacked_frames,
            # TODO: actually pass player
            player=jnp.ones((batch.stacked_frames.shape[0], 1)),
        )
        network_output = network.root_inference(
            params, init_inf_features, is_training=True
        )

        losses = {}

        losses["loss_reward_0"] = vmap(mse)(
            batch.last_reward.take(0, axis=1), network_output.reward
        )
        losses["loss_value_0"] = vmap(mse)(
            batch.value.take(0, axis=1), network_output.value
        )
        losses["loss_action_probs_0"] = vmap(rlax.categorical_cross_entropy)(
            batch.action_probs.take(0, axis=1), network_output.policy_logits
        )

        for i in range(self._num_unroll_steps):
            recurr_inf_features = TransitionInferenceFeatures(
                hidden_state=network_output.hidden_state,
                action=batch.action.take(0, axis=1),
            )
            network_output = network.trans_inference(
                params, recurr_inf_features, is_training=True
            )

            losses[f"loss_reward_{str(i + 1)}"] = vmap(mse)(
                batch.last_reward.take(i + 1, axis=1), network_output.reward
            )
            losses[f"loss_value_{str(i + 1)}"] = vmap(mse)(
                batch.value.take(i + 1, axis=1), network_output.value
            )
            losses[f"loss_action_probs_{str(i + 1)}"] = vmap(
                rlax.categorical_cross_entropy
            )(batch.action_probs.take(i + 1, axis=1), network_output.policy_logits)

        losses["loss_l2"] = jnp.reshape(
            params_l2_loss(params) * self._weight_decay, (1,)
        )

        loss = jnp.sum(jnp.concatenate(tree.flatten(losses)))
        losses["loss"] = loss

        return loss, JAXBoardStepData(
            scalars=tree.map_structure(jnp.mean, losses), histograms={}
        )
