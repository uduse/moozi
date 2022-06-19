import mctx
import jax.numpy as jnp
import jax
import numpy as np
import haiku as hk

from moozi.nn import NNModel, RootFeatures, TransitionFeatures


def make_paritial_recurr_fn(model, state):
    def recurr_fn(params, random_key, action, hidden_state):
        trans_feats = TransitionFeatures(hidden_state, action)
        is_training = False
        nn_output, _ = model.trans_inference(params, state, trans_feats, is_training)
        rnn_output = mctx.RecurrentFnOutput(
            reward=nn_output.reward.squeeze(-1),
            discount=jnp.ones_like(nn_output.reward.squeeze(-1)),
            prior_logits=nn_output.policy_logits,
            value=nn_output.value.squeeze(-1),
        )
        return rnn_output, nn_output.hidden_state

    return recurr_fn


def make_planner(
    num_envs: int,
    dim_actions: int,
    model: NNModel,
    num_simulations: int = 10,
    dirichlet_fraction: float = 0.25,
    dirichlet_alpha: float = 0.3,
    temperature: float = 1.0,
):
    def malloc():
        return {
            "root_value": jnp.zeros(num_envs, dtype=jnp.float32),
            "action": jnp.full(num_envs, fill_value=0, dtype=jnp.int32),
            "action_probs": jnp.full(
                (num_envs, dim_actions), fill_value=0, dtype=jnp.float32
            ),
        }

    def planner(params: hk.Params, state: hk.State, stacked_frames, random_key):
        is_training = False
        random_key, new_key = jax.random.split(random_key, 2)
        root_feats = RootFeatures(
            obs=stacked_frames,
            player=np.zeros((stacked_frames.shape[0]), dtype=np.int32),
        )
        nn_output, _ = model.root_inference(params, state, root_feats, is_training)
        root = mctx.RootFnOutput(
            prior_logits=nn_output.policy_logits,
            value=nn_output.value.squeeze(-1),
            embedding=nn_output.hidden_state,
        )
        nn_output, _ = model.root_inference(params, state, root_feats, is_training)
        policy_output = mctx.muzero_policy(
            params=params,
            rng_key=new_key,
            root=root,
            recurrent_fn=make_paritial_recurr_fn(model, state),
            num_simulations=num_simulations,
            dirichlet_fraction=dirichlet_fraction,
            dirichlet_alpha=dirichlet_alpha,
            temperature=temperature,
        )
        stats = policy_output.search_tree.summary()
        return {
            "action_probs": policy_output.action_weights,
            "root_value": stats.value,
            "random_key": random_key,
        }

    return malloc, planner
