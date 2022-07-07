from typing import Optional, Sequence
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
            # TODO: set discount here
            discount=jnp.full_like(nn_output.reward, fill_value=discount),
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
    output_tree: bool = False,
    discount: float = 1.0,
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
            recurrent_fn=make_paritial_recurr_fn(model, state, discount),
            num_simulations=num_simulations,
            # TODO: max_depth should be the same as num_unroll_steps?
            max_depth=5,
            invalid_actions=invalid_actions,
            dirichlet_fraction=dirichlet_fraction,
            dirichlet_alpha=dirichlet_alpha,
            temperature=temperature,
        )
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


# def make_random_planner(
#     batch_size: int,
#     dim_action: int,
#     output_action: bool = True,
# ) -> Law:
#     def malloc():
#         return {
#             "root_value": jnp.zeros(batch_size, dtype=jnp.float32),
#             "action": jnp.full(batch_size, fill_value=0, dtype=jnp.int32),
#             "action_probs": jnp.full(
#                 (batch_size, dim_action), fill_value=0, dtype=jnp.float32
#             ),
#             "q_values": jnp.full(
#                 (batch_size, dim_action), fill_value=0, dtype=jnp.float32
#             ),
#         }

#     def apply(
#         random_key,
#     ):
#         random_key, new_key = jax.random.split(random_key, 2)
#         action_probs = jax.random.uniform(new_key, shape=(10, dim_action - 1))
#         # action = jnp.argmax(
#         #     , axis=1
#         # )
#         action += 1
#         ret = {
#             "action_probs": policy_output.action_weights,
#             "q_values": stats.qvalues,
#             "root_value": stats.value,
#             "random_key": random_key,
#         }
#         if output_action:
#             ret["action"] = policy_output.action
#         return ret

#     return Law(
#         name="planner",
#         malloc=malloc,
#         apply=link(apply),
#         read=get_keys(apply),
#     )


def convert_tree_to_graph(
    tree: mctx.Tree,
    action_labels: Optional[Sequence[str]] = None,
    batch_index: int = 0,
) -> pygraphviz.AGraph:
    """Converts a search tree into a Graphviz graph.
    Args:
      tree: A `Tree` containing a batch of search data.
      action_labels: Optional labels for edges, defaults to the action index.
      batch_index: Index of the batch element to plot.
    Returns:
      A Graphviz graph representation of `tree`.
    """
    chex.assert_rank(tree.node_values, 2)
    batch_size = tree.node_values.shape[0]
    if action_labels is None:
        action_labels = range(tree.num_actions)
    elif len(action_labels) != tree.num_actions:
        raise ValueError(
            f"action_labels {action_labels} has the wrong number of actions "
            f"({len(action_labels)}). "
            f"Expecting {tree.num_actions}."
        )

    def node_to_str(node_i, reward=0, discount=1):
        return (
            f"{node_i}\n"
            f"Reward: {reward:.2f}\n"
            f"Discount: {discount:.2f}\n"
            f"Value: {tree.node_values[batch_index, node_i]:.2f}\n"
            f"Visits: {tree.node_visits[batch_index, node_i]}\n"
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
            if children_i >= 0:
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
