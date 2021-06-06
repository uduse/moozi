import typing

import chex
import jax
import jax.numpy as jnp
import rlax

import moozi as mz
from moozi.logging import JAXBoardStepData


class LossFn(typing.Protocol):
    def __call__(
        self, network: mz.nn.NeuralNetwork, params: chex.ArrayTree, batch
    ) -> typing.Any:
        r"""Loss function."""


# # TODO: add a TD version of `initial_inference_value_loss`
# def initial_inference_value_loss(network: mz.nn.NeuralNetwork, params, batch):
#     pred_value = network.initial_inference(
#         params, batch.data.observation.observation
#     ).value.squeeze()
#     target_value = batch.data.reward

#     chex.assert_rank([pred_value, target_value], 1)  # shape: (batch_size,)

#     loss_scalar = jnp.mean(jnp.square(pred_value - target_value))
#     return loss_scalar, None


def params_l2_loss(params):
    return 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))


class NStepPriorVanillaPolicyGradientLoss(LossFn):
    def __init__(self, weight_decay: float = 1e-4, entropy_loss: float = 1e-4):
        self._weight_decay = weight_decay
        self._entropy_loss = entropy_loss

    def __call__(
        self,
        network: mz.nn.NeuralNetwork,
        params: chex.ArrayTree,
        batch,
    ) -> typing.Any:
        r"""
        Assume the batch data contains: (s_t, a_t, R_{t:t+n}, D_{t:t+n}, s_{t+N})
        """
        output_t = network.initial_inference(params, batch.data.observation.observation)
        action = batch.data.action
        pg_loss = rlax.policy_gradient_loss(
            logits_t=output_t.policy_logits,
            a_t=action,
            adv_t=batch.data.reward,  # use full return as the target
            w_t=jnp.ones_like(action, dtype=float),
        )
        action_entropy = jnp.mean(rlax.softmax().entropy(logits=output_t.policy_logits))
        chex.assert_rank(action_entropy, 0)
        weight_loss = params_l2_loss(params) * self._weight_decay
        entropy_loss = (
            rlax.entropy_loss(
                output_t.policy_logits, jnp.ones_like(action, dtype=float)
            )
            * self._entropy_loss
        )
        loss = pg_loss + weight_loss + entropy_loss
        return loss, {
            "loss": loss,
            "pg_loss": pg_loss,
            "weight_loss": weight_loss,
            "entropy_loss": entropy_loss,
            "logits": output_t.policy_logits,
            "action_entropy": action_entropy,
        }

        # mz.logging.JAXBoardStepData(
        #     scalars={"loss": pg_loss},
        #     histograms={
        #         "output_logits": output_t.policy_logits,
        #         "action_entropy": action_entropy,
        #     },
        # )


class OneStepAdvantagePolicyGradientLoss(LossFn):
    def __init__(self, weight_decay: float = 1e-4, entropy_loss: float = 1e-4):
        self._weight_decay = weight_decay
        # self._entropy_loss = entropy_loss

    def __call__(
        self,
        network: mz.nn.NeuralNetwork,
        params: chex.ArrayTree,
        batch,
    ) -> typing.Any:
        r"""
        Assume the batch data contains: (s_t, a_t, R_{t:t+n}, D_{t:t+n}, s_{t+N})
        """
        reward = batch.data.reward
        action = batch.data.action
        observation = batch.data.observation.observation
        chex.assert_rank([reward, observation, action], [1, 2, 1])

        output_t = network.initial_inference(params, observation)
        output_tp1 = network.recurrent_inference(params, output_t.hidden_state, action)

        # compute loss
        v_val = output_t.value
        q_val = output_tp1.value
        v_loss = jnp.mean(rlax.l2_loss(v_val, reward))
        q_loss = jnp.mean(rlax.l2_loss(q_val, reward))
        adv = q_val - v_val
        pg_loss = rlax.policy_gradient_loss(
            logits_t=output_t.policy_logits,
            a_t=action,
            adv_t=adv,
            w_t=jnp.ones_like(action, dtype=float),
        )
        l2_loss = params_l2_loss(params) * self._weight_decay
        loss = pg_loss + l2_loss + v_loss + q_loss
        chex.assert_rank([pg_loss, l2_loss, v_loss, q_loss, loss], 0)

        # compute other info
        action_entropy = jnp.mean(rlax.softmax().entropy(logits=output_t.policy_logits))
        chex.assert_rank(action_entropy, 0)
        return loss, JAXBoardStepData(
            scalars={
                "loss": loss,
                "v_loss": v_loss,
                "q_loss": q_loss,
                "pg_loss": pg_loss,
                "l2_loss": l2_loss,
                "action_entropy": action_entropy,
            },
            histograms={
                "v": output_t.value,
                "q": output_tp1.value,
                "logits": output_t.policy_logits,
            },
        )
