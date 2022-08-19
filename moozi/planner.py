from typing import Optional, Sequence
from flax import struct
import mctx
import chex
import jax.numpy as jnp
import jax
import numpy as np
import haiku as hk
import pygraphviz
from moozi.core.link import link

from moozi.nn import NNModel, RootFeatures, TransitionFeatures
from moozi.laws import Law, get_keys


def make_paritial_recurr_fn(model, state, discount):
    def recurr_fn(params, random_key, action, hidden_state):
        trans_feats = TransitionFeatures(hidden_state, action)
        is_training = False
        nn_output, _ = model.trans_inference(params, state, trans_feats, is_training)
        chex.assert_shape(nn_output.reward, (None,))
        chex.assert_shape(nn_output.value, (None,))
        rnn_output = mctx.RecurrentFnOutput(
            reward=nn_output.reward,
            discount=jnp.full_like(nn_output.reward, fill_value=discount),
            prior_logits=nn_output.policy_logits,
            value=nn_output.value,
        )
        return rnn_output, nn_output.hidden_state

    return recurr_fn


def qtransform_by_parent_and_siblings_inherit(
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
                qtransform=qtransform_by_parent_and_siblings_inherit,
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


def convert_tree_to_graph(
    tree: mctx.Tree,
    action_labels: Optional[Sequence[str]] = None,
    batch_index: int = 0,
    show_only_expanded: bool = True,
) -> pygraphviz.AGraph:
    """Converts a search tree into a Graphviz graph.
    Args:
      tree: A `Tree` containing a batch of search data.
      action_labels: Optional labels for edges, defaults to the action index.
      batch_index: Index of the batch element to plot.
    Returns:
      A Graphviz graph representation of `tree`.

    Copy-pasted from mctx library examples.
    https://github.com/deepmind/mctx/blob/main/examples/visualization_demo.py
    """
    chex.assert_rank(tree.node_values, 2)
    batch_size = tree.node_values.shape[0]
    if action_labels is None:
        action_labels = list(map(str, range(tree.num_actions)))
    elif len(action_labels) != tree.num_actions:
        raise ValueError(
            f"action_labels {action_labels} has the wrong number of actions "
            f"({len(action_labels)}). "
            f"Expecting {tree.num_actions}."
        )

    def node_to_str(node_i, reward=0, discount=1):
        return (
            f"{node_i}\n"
            f"R: {reward:.2f}\n"
            f"d: {discount:.2f}\n"
            f"V: {tree.node_values[batch_index, node_i]:.2f}\n"
            f"N: {tree.node_visits[batch_index, node_i]}\n"
        )

    def edge_to_str(node_i, a_i):
        node_index = jnp.full([batch_size], node_i)
        probs = jax.nn.softmax(tree.children_prior_logits[batch_index, node_i])
        return (
            f"{action_labels[a_i]}\n"
            f"Q: {tree.qvalues(node_index)[batch_index, a_i]:.2f}\n"
            f"p: {probs[a_i]:.2f}\n"
        )

    graph = pygraphviz.AGraph(directed=True)

    # Add root
    graph.add_node(0, label=node_to_str(node_i=0), color="green")
    # Add all other nodes and connect them up.
    for node_i in range(tree.num_simulations):
        for a_i in range(tree.num_actions):
            # Index of children, or -1 if not expanded
            children_i = tree.children_index[batch_index, node_i, a_i]
            if show_only_expanded:
                to_show = children_i >= 0
            else:
                to_show = True
            if to_show:
                graph.add_node(
                    children_i,
                    label=node_to_str(
                        node_i=children_i,
                        reward=tree.children_rewards[batch_index, node_i, a_i],
                        discount=tree.children_discounts[batch_index, node_i, a_i],
                    ),
                    color="red",
                )
                graph.add_edge(node_i, children_i, label=edge_to_str(node_i, a_i))

    return graph


class Planner(struct.PyTreeNode):
    batch_size: int
    dim_action: int
    model: NNModel = struct.field(pytree_node=False)
    discount: float = 1.0
    num_unroll_steps: int = struct.field(pytree_node=False, default=5)
    num_simulations: int = struct.field(pytree_node=False, default=10)
    limit_depth: bool = struct.field(pytree_node=False, default=True)
    use_gumbel: bool = struct.field(pytree_node=False, default=True)

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

    def run(self, feed: "PlannerFeed") -> "PlannerOut":
        is_training = False
        nn_output, _ = self.model.root_inference(
            feed.params, feed.state, feed.root_feats, is_training
        )
        root = mctx.RootFnOutput(
            prior_logits=nn_output.policy_logits,
            value=nn_output.value,
            embedding=nn_output.hidden_state,
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
                max_depth=self.num_unroll_steps if self.limit_depth else None,
                invalid_actions=invalid_actions,
                max_num_considered_actions=16,
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
                max_depth=self.num_unroll_steps if self.limit_depth else None,
                invalid_actions=invalid_actions,
                qtransform=qtransform_by_parent_and_siblings_inherit,
            )

        action = mctx_out.action
        stats = mctx_out.search_tree.summary()
        prior_probs = jax.nn.softmax(nn_output.policy_logits)
        visit_counts = stats.visit_counts
        action_probs = mctx_out.action_weights
        q_values = stats.qvalues
        root_value = stats.value
        # if self.output_tree:
        tree = mctx_out.search_tree
        # else:
        #     tree = None

        return self.PlannerOut(
            action=action,
            action_probs=action_probs,
            tree=tree,
            prior_probs=prior_probs,
            visit_counts=visit_counts,
            q_values=q_values,
            root_value=root_value,
        )
