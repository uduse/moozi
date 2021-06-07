# %%
import graphviz
import haiku as hk
import jax
import jax.numpy as jnp
import moozi as mz
from pathlib import Path

# %%
project_root_dir = Path(__file__).parent.parent
vis_dir = project_root_dir / "vis"

# %%
dim_batch = 32
dim_image = 35
dim_repr = 2
dim_action = 3
network = mz.nn.get_network(
    mz.nn.NeuralNetworkSpec(
        dim_image=dim_image,
        dim_repr=dim_repr,
        dim_action=dim_action,
        repr_net_sizes=(2,),
        pred_net_sizes=(2,),
        dyna_net_sizes=(2,),
    )
)
key = jax.random.PRNGKey(0)
params = network.init(key)
image = jnp.ones((dim_batch, dim_image))
action = jnp.ones(dim_batch)

# %%
ini_inf = hk.experimental.to_dot(network.initial_inference)(params, image)
ini_inf = graphviz.Source(ini_inf)
ini_inf.render(vis_dir / "mlp_initial_inference", cleanup=True)

# %%
repr_ = network.initial_inference(params, image).hidden_state
recur_inf = hk.experimental.to_dot(network.recurrent_inference)(params, repr_, action)
recur_inf = graphviz.Source(recur_inf)
recur_inf.render(vis_dir / "mlp_recurrent_inference", cleanup=True)

# %%
