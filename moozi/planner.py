from typing import Optional, Sequence, Union
from flax import struct
import mctx
import chex
import jax.numpy as jnp
import jax
import numpy as np
import haiku as hk
import pygraphviz
from moozi import BASE_PLAYER
from moozi.core.link import link

from moozi.nn import NNModel, RootFeatures, TransitionFeatures
from moozi.laws import Law, get_keys


def _view_from_player(scalar: chex.Array, to_play: chex.Array):
    return jax.lax.select(to_play == BASE_PLAYER, scalar, -scalar)


def _next_player(player: chex.Array):
    return jnp.logical_not(player).astype(player.dtype)


def make_paritial_recurr_fn(model: NNModel, state: hk.State, discount: float):
    def recurr_fn(params, random_key, action, embedding):
        hidden_state, to_play = embedding
        trans_feats = TransitionFeatures(hidden_state, action)
        is_training = False
        nn_output, _ = model.trans_inference(params, state, trans_feats, is_training)
        chex.assert_shape(nn_output.reward, (None,))
        chex.assert_shape(nn_output.value, (None,))
        rnn_output = mctx.RecurrentFnOutput(
            reward=_view_from_player(nn_output.reward, to_play),
            discount=jnp.full_like(nn_output.reward, fill_value=discount),
            prior_logits=nn_output.policy_logits,
            value=_view_from_player(nn_output.value, to_play),
        )
        return rnn_output, (nn_output.hidden_state, _next_player(to_play))

    return recurr_fn


def qtransform_by_inheritance(
    tree: mctx.Tree,
    node_index: chex.Numeric,
    *,
    epsilon: chex.Numeric = 1e-8,
) -> chex.Array:
    """Returns qvalues normalized by min, max over V(node) and qvalues.

    Args:
      tree: _unbatched_ MCTS tree state.
      node_index: scalar index of the parent node.
      epsilon: the minimum denominator for the normalization.

    Returns:
      Q-values normalized to be from the [0, 1] interval. The unvisited actions
      will have zero Q-value. Shape `[num_actions]`.
    """
    chex.assert_shape(node_index, ())
    qvalues = tree.qvalues(node_index)
    visit_counts = tree.children_visits[node_index]
    chex.assert_rank([qvalues, visit_counts, node_index], [1, 1, 0])
    node_value = tree.node_values[node_index]
    safe_qvalues = jnp.where(visit_counts > 0, qvalues, node_value)
    chex.assert_equal_shape([safe_qvalues, qvalues])
    min_value = jnp.minimum(node_value, jnp.min(safe_qvalues, axis=-1))
    max_value = jnp.maximum(node_value, jnp.max(safe_qvalues, axis=-1))

    completed_by_parent = jnp.where(visit_counts > 0, qvalues, node_value)
    normalized = (completed_by_parent - min_value) / (
        jnp.maximum(max_value - min_value, epsilon)
    )
    chex.assert_equal_shape([normalized, qvalues])
    return normalized


def make_planner(
    batch_size: int,
    dim_action: int,
    model: NNModel,
    discount: float = 1.0,
    num_unroll_steps: int = 5,
    num_simulations: int = 10,
    output_action: bool = True,
    output_tree: bool = False,
    policy_type: str = "mcts",
    limit_depth: bool = False,
    kwargs: dict = {},
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
        root_feats = RootFeatures(
            obs=obs, to_play=np.zeros((batch_size,), dtype=jnp.int32)
        )
        nn_output, _ = model.root_inference(params, state, root_feats, is_training)
        root = mctx.RootFnOutput(
            prior_logits=nn_output.policy_logits,
            value=nn_output.value,
            embedding=nn_output.hidden_state,
        )

        # the planning of the last step will be overwritten by the target
        # creating process so it's okay that we always mast the no-op action
        invalid_actions = np.zeros((batch_size, dim_action))
        invalid_actions[:, 0] = 1
        random_key, search_key = jax.random.split(random_key, 2)

        if policy_type == "mcts":
            policy_output = mctx.muzero_policy(
                params=params,
                rng_key=search_key,
                root=root,
                recurrent_fn=make_paritial_recurr_fn(model, state, discount),
                num_simulations=num_simulations,
                max_depth=num_unroll_steps if limit_depth else None,
                invalid_actions=invalid_actions,
                qtransform=qtransform_by_inheritance,
                **kwargs,
            )

        elif policy_type == "gumbel":
            policy_output = mctx.gumbel_muzero_policy(
                params=params,
                rng_key=search_key,
                root=root,
                recurrent_fn=make_paritial_recurr_fn(model, state, discount),
                num_simulations=num_simulations,
                invalid_actions=invalid_actions,
                max_depth=num_unroll_steps if limit_depth else None,
                **kwargs,
            )
        else:
            # TODO: add prior policy
            raise ValueError
        stats = policy_output.search_tree.summary()
        ret = {
            "prior_probs": jax.nn.softmax(nn_output.policy_logits),
            "visit_counts": stats.visit_counts,
            "action_probs": policy_output.action_weights,
            "q_values": stats.qvalues,
            "root_value": stats.value,
            "random_key": random_key,
        }
        if output_action:
            # doesn't output action to reanalyze
            ret["action"] = policy_output.action
        if output_tree:
            ret["tree"] = policy_output.search_tree
        return ret

    return Law(
        name="planner",
        malloc=malloc,
        apply=link(apply),
        read=get_keys(apply),
    )


class PlannerFeed(struct.PyTreeNode):
    params: hk.Params
    state: hk.State
    root_feats: RootFeatures
    legal_actions: chex.Array
    random_key: chex.PRNGKey


class PlannerOut(struct.PyTreeNode):
    action: Optional[chex.ArrayDevice]
    action_probs: chex.Array
    tree: Optional[mctx.Tree]
    prior_probs: chex.Array
    visit_counts: chex.Array
    q_values: chex.Array
    root_value: chex.Array


class Planner(struct.PyTreeNode):
    batch_size: int
    dim_action: int
    model: NNModel = struct.field(pytree_node=False)
    discount: float = 1.0
    # num_unroll_steps: int = struct.field(pytree_node=False, default=5)
    num_simulations: int = struct.field(pytree_node=False, default=10)
    max_depth: Optional[int] = struct.field(pytree_node=False, default=None)
    use_gumbel: bool = struct.field(pytree_node=False, default=False)
    kwargs: dict = struct.field(default_factory=dict)

    def run(self, feed: "PlannerFeed") -> "PlannerOut":
        is_training = False
        nn_output, _ = self.model.root_inference(
            feed.params, feed.state, feed.root_feats, is_training
        )
        root = mctx.RootFnOutput(
            prior_logits=nn_output.policy_logits,
            value=_view_from_player(nn_output.value, feed.root_feats.to_play),
            embedding=(nn_output.hidden_state, feed.root_feats.to_play),
        )
        invalid_actions = jnp.logical_not(feed.legal_actions)

        if self.use_gumbel:
            mctx_out = mctx.gumbel_muzero_policy(
                params=feed.params,
                rng_key=feed.random_key,
                root=root,
                recurrent_fn=make_paritial_recurr_fn(
                    self.model, feed.state, self.discount
                ),
                num_simulations=self.num_simulations,
                max_depth=self.max_depth,
                invalid_actions=invalid_actions,
                max_num_considered_actions=16,
                **self.kwargs
            )
        else:
            mctx_out = mctx.muzero_policy(
                params=feed.params,
                rng_key=feed.random_key,
                root=root,
                recurrent_fn=make_paritial_recurr_fn(
                    self.model, feed.state, self.discount
                ),
                num_simulations=self.num_simulations,
                max_depth=self.max_depth,
                invalid_actions=invalid_actions,
                **self.kwargs
            )

        action = mctx_out.action
        stats = mctx_out.search_tree.summary()
        prior_probs = jax.nn.softmax(nn_output.policy_logits)
        visit_counts = stats.visit_counts
        action_probs = mctx_out.action_weights
        q_values = stats.qvalues
        root_value = stats.value
        tree = mctx_out.search_tree

        return PlannerOut(
            action=action,
            action_probs=action_probs,
            tree=tree,
            prior_probs=prior_probs,
            visit_counts=visit_counts,
            q_values=q_values,
            root_value=root_value,
        )
