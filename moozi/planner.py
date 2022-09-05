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
    num_players: int
    model: NNModel = struct.field(pytree_node=False)

    discount: float = 1.0
    num_simulations: int = 10
    max_depth: Optional[int] = None
    search_type: str = "muzero"
    kwargs: dict = struct.field(default_factory=dict)

    def run(self, feed: "PlannerFeed") -> "PlannerOut":
        is_training = False
        nn_output, _ = self.model.root_inference(
            feed.params, feed.state, feed.root_feats, is_training
        )
        root = mctx.RootFnOutput(
            prior_logits=nn_output.policy_logits,
            value=self._view_from_player(nn_output.value, feed.root_feats.to_play),
            embedding=(nn_output.hidden_state, feed.root_feats.to_play),
        )
        invalid_actions = jnp.logical_not(feed.legal_actions)

        if self.search_type == "muzero":
            mctx_out = mctx.muzero_policy(
                params=feed.params,
                rng_key=feed.random_key,
                root=root,
                recurrent_fn=self.make_paritial_recurr_fn(self.model, feed.state),
                num_simulations=self.num_simulations,
                max_depth=self.max_depth,
                invalid_actions=invalid_actions,
                **self.kwargs,
            )
        elif self.search_type == "gumbel_muzero":
            mctx_out = mctx.gumbel_muzero_policy(
                params=feed.params,
                rng_key=feed.random_key,
                root=root,
                recurrent_fn=self.make_paritial_recurr_fn(self.model, feed.state),
                num_simulations=self.num_simulations,
                max_depth=self.max_depth,
                invalid_actions=invalid_actions,
                **self.kwargs,
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

    def _view_from_player(self, scalar: chex.Array, player: chex.Array) -> chex.Array:
        return jax.lax.select(player == BASE_PLAYER, scalar, -scalar)

    def _next_player(self, player: chex.Array):
        if self.num_players == 2:
            return jnp.logical_not(player).astype(player.dtype)
        elif self.num_players == 1:
            return player
        else:
            raise NotImplementedError

    def make_paritial_recurr_fn(
        self,
        model: NNModel,
        state: hk.State,
    ):
        def recurr_fn(params, random_key, action, embedding):
            hidden_state, prev_player = embedding
            curr_player = self._next_player(prev_player)
            trans_feats = TransitionFeatures(hidden_state, action)
            is_training = False
            nn_output, _ = model.trans_inference(
                params, state, trans_feats, is_training
            )
            chex.assert_shape(nn_output.reward, (None,))
            chex.assert_shape(nn_output.value, (None,))
            search_discount = jnp.full_like(
                nn_output.reward, fill_value=self._get_search_discount()
            )
            rnn_output = mctx.RecurrentFnOutput(
                reward=self._view_from_player(nn_output.reward, prev_player),
                discount=search_discount,
                prior_logits=nn_output.policy_logits,
                value=self._view_from_player(nn_output.value, curr_player),
            )
            return rnn_output, (nn_output.hidden_state, curr_player)

        return recurr_fn

    def _get_search_discount(self):
        if self.num_players == 1:
            return self.discount
        elif self.num_players == 2:
            # Use negative discount for two-player zero-sum games
            return -self.discount
        else:
            raise NotImplementedError
