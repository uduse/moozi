import typing
import jax.numpy as jnp
import chex

import moozi as mz


class LossExtra(typing.NamedTuple):
    metrics: typing.Dict[str, jnp.DeviceArray]


def initial_inference_value_loss(network: mz.nn.NeuralNetwork, params, batch):
    pred_value = network.initial_inference(
        params, batch.data.observation.observation
    ).value.squeeze()
    target_value = batch.data.reward

    chex.assert_rank([pred_value, target_value], 2)  # (batch, dim_image)

    loss_scalar = jnp.mean(jnp.square(pred_value - target_value))
    extra = LossExtra({})
    return loss_scalar, extra


# TODO: add a TD version of `initial_inference_value_loss`
