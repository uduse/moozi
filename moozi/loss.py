import functools
from typing import Tuple, Any

import chex
import jax
import jax.numpy as jnp
import rlax
from acme.jax.utils import add_batch_dim
from jax import vmap

import moozi as mz
from moozi.logging import JAXBoardStepData


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


class OneStepAdvantagePolicyGradientLoss(LossFn):
    def __init__(self, weight_decay: float = 1e-4, entropy_reg: float = 1e-1):
        self._weight_decay = weight_decay
        self._entropy_reg = entropy_reg

    def __call__(
        self,
        network: mz.nn.NeuralNetwork,
        params: chex.ArrayTree,
        batch: mz.replay.TrainTarget,
    ) -> Tuple[chex.ArrayDevice, Any]:
        value = jnp.take(batch.value, 0, axis=-1)
        action = jnp.take(batch.action, 0, axis=-1)
        stacked_frames = batch.stacked_frames

        chex.assert_rank([value, stacked_frames, action], [1, 3, 1])
        chex.assert_equal_shape_prefix([value, stacked_frames, action], 1)

        output_t = network.initial_inference(params, stacked_frames)
        output_tp1 = network.recurrent_inference(params, output_t.hidden_state, action)

        # compute loss
        v_val = output_t.value
        q_val = output_tp1.value
        v_loss = jnp.mean(rlax.l2_loss(v_val, value))
        q_loss = jnp.mean(rlax.l2_loss(q_val, value))
        adv = q_val - v_val
        pg_loss = rlax.policy_gradient_loss(
            logits_t=output_t.policy_logits,
            a_t=action,
            adv_t=adv,
            w_t=jnp.ones_like(action, dtype=float),
        )
        l2_loss = params_l2_loss(params) * self._weight_decay
        raw_entropy = rlax.softmax().entropy(logits=output_t.policy_logits)
        entropy_loss = -jnp.mean(raw_entropy) * self._entropy_reg
        loss = pg_loss + l2_loss + v_loss + q_loss + entropy_loss
        chex.assert_rank([loss, pg_loss, l2_loss, v_loss, q_loss, entropy_loss], 0)

        return loss, JAXBoardStepData(
            scalars={
                "loss": loss,
                "v_loss": v_loss,
                "q_loss": q_loss,
                "pg_loss": pg_loss,
                "l2_loss": l2_loss,
                "entropy_loss": entropy_loss,
            },
            histograms={
                "v": output_t.value,
                "q": output_tp1.value,
                "logits": output_t.policy_logits,
            },
        )


def mse(a, b):
    return jnp.mean((a - b) ** 2)


class MCTSLoss(LossFn):
    def __init__(self, num_unroll_steps, weight_decay=1e-4):
        self._num_unroll_steps = num_unroll_steps
        self._weight_decay = weight_decay

    def __call__(
        self,
        network: mz.nn.NeuralNetwork,
        params: chex.ArrayTree,
        batch: mz.replay.TrainTarget,
    ) -> Tuple[chex.ArrayDevice, Any]:

        chex.assert_rank(
            [
                batch.stacked_frames,
                batch.action,
                batch.value,
                batch.last_reward,
                batch.child_visits,
            ],
            [3, 2, 2, 2, 3],
        )

        loss = jnp.array([0])
        network_output = network.initial_inference(params, batch.stacked_frames)
        loss += vmap(mse)(batch.last_reward.take(0, axis=1), network_output.reward)
        loss += vmap(mse)(batch.value.take(0, axis=1), network_output.value)
        loss += vmap(rlax.categorical_cross_entropy)(
            batch.child_visits.take(0, axis=1), network_output.policy_logits
        )

        for i in range(self._num_unroll_steps):
            step_loss = 0
            network_output = network.recurrent_inference(
                params, network_output.hidden_state, batch.action.take(0, axis=1)
            )

            step_loss += vmap(mse)(
                batch.last_reward.take(i + 1, axis=1), network_output.reward
            )
            step_loss += vmap(mse)(
                batch.value.take(i + 1, axis=1), network_output.value
            )
            step_loss += vmap(rlax.categorical_cross_entropy)(
                batch.child_visits.take(i + 1, axis=1), network_output.policy_logits
            )

            invalid_action_mask = batch.action.take(0, axis=1) == -1
            step_loss *= invalid_action_mask
            loss += step_loss

        loss += params_l2_loss(params) * self._weight_decay
        loss = jnp.mean(loss)

        return loss, JAXBoardStepData(scalars={"loss": loss}, histograms={})
