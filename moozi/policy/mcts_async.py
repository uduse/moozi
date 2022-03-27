from dataclasses import InitVar, dataclass
import copy
from typing import Awaitable, Callable

import inspect
from jax._src.numpy.lax_numpy import isin
import jax.numpy as jnp
import numpy as np
from moozi import link
from moozi.batching_layer import BatchingClient
from moozi.nn import (
    RootFeatures,
    NNModel,
    NNSpec,
    NNOutput,
    TransitionFeatures,
)
from moozi.policy.mcts_core import (
    Node,
    SearchStrategy,
    get_next_player,
    get_prev_player,
    reorient,
)
from moozi import PolicyFeed, BASE_PLAYER
from moozi.utils import as_coroutine


@dataclass
class MCTSAsync:
    root_inf_fn: Callable[[RootFeatures], Awaitable[NNOutput]]
    trans_inf_fn: Callable[[TransitionFeatures], Awaitable[NNOutput]]
    dim_action: int

    strategy: SearchStrategy = SearchStrategy.TWO_PLAYER
    num_simulations: int = 10
    allow_all_actions_mask: np.ndarray = None
    discount: float = 1.0

    def __post_init__(self):
        self.allow_all_actions_mask = np.ones((self.dim_action,), dtype=np.int32)

    async def run(self, feed: PolicyFeed) -> Node:
        root = await self.get_root(feed)
        root.add_exploration_noise()

        for _ in range(self.num_simulations):
            await self.simulate_once(root)

        return root

    async def get_root(self, feed: PolicyFeed) -> Node:
        root_feats = RootFeatures(
            obs=feed.stacked_frames, player=np.array(feed.to_play)
        )
        root_nn_output = await self.root_inf_fn(root_feats)
        root = Node(0, player=feed.to_play, name="s")
        next_player = get_next_player(self.strategy, feed.to_play)
        root.expand_node(
            hidden_state=root_nn_output.hidden_state,
            reward=0.0,
            policy_logits=root_nn_output.policy_logits,
            legal_actions_mask=feed.legal_actions_mask,
            next_player=next_player,
        )
        value = float(root_nn_output.value)
        root.backpropagate(value=value, discount=self.discount)
        return root

    async def simulate_once(self, root: Node):
        action, leaf = root.select_leaf(discount=self.discount)
        assert leaf.parent

        trans_feats = TransitionFeatures(
            hidden_state=leaf.parent.hidden_state, action=np.array(action)
        )
        leaf_nn_output = await self.trans_inf_fn(trans_feats)

        reward = float(leaf_nn_output.reward)
        value = float(leaf_nn_output.value)

        leaf.expand_node(
            hidden_state=leaf_nn_output.hidden_state,
            reward=reward,
            policy_logits=leaf_nn_output.policy_logits,
            legal_actions_mask=self.allow_all_actions_mask,
            next_player=get_next_player(self.strategy, leaf.player),
        )

        leaf.backpropagate(value=value, discount=self.discount)


@link
async def planner_law(
    is_last,
    legal_actions_mask,
    policy_feed,
    root_inf_fn,
    trans_inf_fn,
    num_simulations,
):
    if not is_last:
        mcts = MCTSAsync(
            root_inf_fn=root_inf_fn,
            trans_inf_fn=trans_inf_fn,
            dim_action=legal_actions_mask.size,
            num_simulations=num_simulations,
        )
        mcts_root = await mcts.run(policy_feed)
        action_probs = mcts_root.get_children_visit_counts_as_probs(
            dim_action=mcts.dim_action
        )

        return dict(
            action_probs=action_probs,
            mcts_root=copy.deepcopy(mcts_root),
        )


@link
@dataclass
class Planner:
    num_simulations: int

    async def __call__(
        self,
        is_last,
        legal_actions_mask,
        policy_feed,
        root_inf_fn,
        trans_inf_fn,
    ):
        if not is_last:
            mcts = MCTSAsync(
                root_inf_fn=root_inf_fn,
                trans_inf_fn=trans_inf_fn,
                dim_action=legal_actions_mask.size,
                num_simulations=self.num_simulations,
            )
            mcts_root = await mcts.run(policy_feed)
            action_probs = mcts_root.get_children_visit_counts_as_probs(
                dim_action=mcts.dim_action
            )

            return dict(
                action_probs=action_probs,
                mcts_root=copy.deepcopy(mcts_root),
            )


def sample_action(action_probs, temperature=1.0):
    log_probs = np.log(np.clip(action_probs, 1e-10, None)) / temperature
    action_probs = np.exp(log_probs) / np.sum(np.exp(log_probs))
    return np.random.choice(np.arange(len(action_probs)), p=action_probs)


@link
@dataclass
class ActionSamplerLaw:
    temperature: float = 1.0

    def __call__(self, action_probs):
        action = sample_action(action_probs, temperature=self.temperature)
        return dict(action=action)


def temp(arr, temperature=1.0):
    log = np.log(arr) / temperature
    return np.exp(log) / np.sum(np.exp(log))
