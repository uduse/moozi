import typing
import jax.numpy as jnp

import moozi as mz

class LossExtra(typing.NamedTuple):
    metrics: typing.Dict[str, jnp.DeviceArray]


def initial_inference_value_loss(network: mz.nn.NeuralNetwork, params, batch):
    inf_out = network.initial_inference(params, batch.data.observation.observation)
    loss_scalar = jnp.mean(jnp.square(inf_out.value - batch.data.reward))
    extra = LossExtra({})
    return loss_scalar, extra


# TODO: add a TD version of `initial_inference_value_loss`
