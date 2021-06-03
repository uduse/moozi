import typing
import jax
import jax.numpy as jnp
import chex
import rlax

import moozi as mz


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


class NStepPriorVanillaPolicyGradientLoss(LossFn):
    def __call__(
        self, network: mz.nn.NeuralNetwork, params: chex.ArrayTree, batch
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
        l2_loss = rlax.l2_loss(params)
        loss = pg_loss + l2_loss
        return loss, {
            "loss": loss,
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


# def n_step_prior_adv_policy_gradient_loss(network: mz.nn.NeuralNetwork, params, batch):
#     r"""
#     Assume the batch data contains: (s_t, a_t, R_{t:t+n}, D_{t:t+n}, s_{t+N})
#     """
#     output_t = network.initial_inference(params, batch.data.observation.observation)
#     action = batch.data.action
#     output_t_next = network.recurrent_inference(params, output_t.hidden_state, action)
#     pg_loss = rlax.policy_gradient_loss(
#         logits_t=output_t_next.policy_logits,
#         a_t=action,
#         # adv_t=output_t_next.value - output_t.value,  # adv = q - v
#         adv_t=output_t_next.value,  # adv = q - v
#         w_t=jnp.ones_like(action, dtype=float),
#     )
#     v_td_loss = jax.vmap(rlax.td_learning)(
#         v_tm1=output_t.value,
#         r_t=batch.data.reward,
#         discount_t=batch.data.discount,
#         v_t=output_t_next.value,
#     )
#     return jnp.mean(v_td_loss + pg_loss), None
