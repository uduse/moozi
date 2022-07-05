import mctx
import jax.numpy as jnp
import jax
import numpy as np
import haiku as hk
from moozi.core.link import link

from moozi.nn import NNModel, RootFeatures, TransitionFeatures
from moozi.laws import Law, get_keys


def make_paritial_recurr_fn(model, state):
    def recurr_fn(params, random_key, action, hidden_state):
        trans_feats = TransitionFeatures(hidden_state, action)
        is_training = False
        nn_output, _ = model.trans_inference(params, state, trans_feats, is_training)
        rnn_output = mctx.RecurrentFnOutput(
            reward=nn_output.reward,
            # TODO: set discount here
            discount=jnp.full_like(nn_output.reward, fill_value=0.99),
            prior_logits=nn_output.policy_logits,
            value=nn_output.value,
        )
        return rnn_output, nn_output.hidden_state

    return recurr_fn


def make_planner(
    batch_size: int,
    dim_action: int,
    model: NNModel,
    num_simulations: int = 10,
    dirichlet_fraction: float = 0.25,
    dirichlet_alpha: float = 0.3,
    temperature: float = 1.0,
    output_action: bool = True,
) -> Law:
    def malloc():
        return {
            "root_value": jnp.zeros(batch_size, dtype=jnp.float32),
            "action": jnp.full(batch_size, fill_value=0, dtype=jnp.int32),
            "action_probs": jnp.full(
                (batch_size, dim_action), fill_value=0, dtype=jnp.float32
            ),
            "q_values": jnp.full(
                (batch_size, dim_action), fill_value=0, dtype=jnp.float32
            ),
        }

    def apply(
        params: hk.Params,
        state: hk.State,
        obs,
        random_key,
    ):
        is_training = False
        batch_size = obs.shape[0]
        random_key, new_key = jax.random.split(random_key, 2)
        root_feats = RootFeatures(
            obs=obs, player=np.zeros((batch_size,), dtype=np.int32)
        )
        nn_output, _ = model.root_inference(params, state, root_feats, is_training)
        root = mctx.RootFnOutput(
            prior_logits=nn_output.policy_logits,
            value=nn_output.value,
            embedding=nn_output.hidden_state,
        )
        # using numpy because it's constant
        invalid_actions = np.zeros((batch_size, dim_action))
        invalid_actions[:, 0] = 1
        policy_output = mctx.muzero_policy(
            params=params,
            rng_key=new_key,
            root=root,
            recurrent_fn=make_paritial_recurr_fn(model, state),
            num_simulations=num_simulations,
            invalid_actions=invalid_actions,
            dirichlet_fraction=dirichlet_fraction,
            dirichlet_alpha=dirichlet_alpha,
            temperature=temperature,
        )
        stats = policy_output.search_tree.summary()
        ret = {
            "action_probs": policy_output.action_weights,
            "q_values": stats.qvalues,
            "root_value": stats.value,
            "random_key": random_key,
        }
        # output action to act
        # doesn't output action to reanalyze
        if output_action:
            ret["action"] = policy_output.action
        return ret

    return Law(
        name="planner",
        malloc=malloc,
        apply=link(apply),
        read=get_keys(apply),
    )
