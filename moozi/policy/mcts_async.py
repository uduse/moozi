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
    InitialInferenceFeatures,
    NeuralNetwork,
    NeuralNetworkSpec,
    NNOutput,
    RecurrentInferenceFeatures,
    get_network,
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
    dim_action: int

    init_inf_fn: Callable[[InitialInferenceFeatures], Awaitable[NNOutput]]
    recurr_inf_fn: Callable[[RecurrentInferenceFeatures], Awaitable[NNOutput]]
    num_simulations: int = 1
    all_actions_mask: np.ndarray = None
    discount: float = 1.0

    strategy: SearchStrategy = SearchStrategy.TWO_PLAYER

    def __post_init__(self):
        self.all_actions_mask = np.ones((self.dim_action,), dtype=np.int32)
        if not inspect.iscoroutinefunction(self.init_inf_fn):
            self.init_inf_fn = as_coroutine(self.init_inf_fn)
        if not inspect.iscoroutinefunction(self.recurr_inf_fn):
            self.recurr_inf_fn = as_coroutine(self.recurr_inf_fn)

    async def run(self, feed: PolicyFeed) -> Node:
        root = await self.get_root(feed)
        root.add_exploration_noise()

        for _ in range(self.num_simulations):
            await self.simulate_once(root)

        return root

    async def get_root(self, feed: PolicyFeed) -> Node:
        init_inf_features = InitialInferenceFeatures(
            stacked_frames=feed.stacked_frames, player=np.array(feed.to_play)
        )
        root_nn_output = await self.init_inf_fn(init_inf_features)
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
        action, leaf = root.select_leaf()
        assert leaf.parent

        recurr_inf_features = RecurrentInferenceFeatures(
            hidden_state=leaf.parent.hidden_state, action=np.array(action)
        )
        leaf_nn_output = await self.recurr_inf_fn(recurr_inf_features)

        reward = float(leaf_nn_output.reward)
        value = float(leaf_nn_output.value)

        leaf.expand_node(
            hidden_state=leaf_nn_output.hidden_state,
            reward=reward,
            policy_logits=leaf_nn_output.policy_logits,
            legal_actions_mask=self.all_actions_mask,
            next_player=get_next_player(self.strategy, leaf.player),
        )

        leaf.backpropagate(value=value, discount=self.discount)


def make_async_planner_law(
    init_inf_fn, recurr_inf_fn, dim_actions, num_simulations=10, include_tree=False
):
    mcts = MCTSAsync(
        init_inf_fn=init_inf_fn,
        recurr_inf_fn=recurr_inf_fn,
        num_simulations=num_simulations,
        dim_action=dim_actions,
    )

    @link
    async def planner(is_last, policy_feed):
        if not is_last:
            mcts_root = await mcts.run(policy_feed)
            action, _ = mcts_root.select_child()

            action_probs = np.zeros((dim_actions,), dtype=np.float32)
            for a, visit_count in mcts_root.get_children_visit_counts().items():
                action_probs[a] = visit_count
            action_probs /= np.sum(action_probs)

            if policy_feed.legal_actions_mask[action] < 1:
                raise ValueError("Illegal action")

            if include_tree:
                return dict(
                    action=action,
                    action_probs=action_probs,
                    mcts_root=copy.deepcopy(mcts_root),
                )
            else:
                return dict(action=action, action_probs=action_probs)

    return planner


def make_async_planner_law_v2(
    init_inf_fn, recurr_inf_fn, dim_actions, num_simulations=10
):
    mcts = MCTSAsync(
        init_inf_fn=init_inf_fn,
        recurr_inf_fn=recurr_inf_fn,
        num_simulations=num_simulations,
        dim_action=dim_actions,
    )

    @link
    async def planner(is_last, policy_feed):
        if not is_last:
            mcts_root = await mcts.run(policy_feed)
            action_probs = mcts_root.get_children_visit_counts_as_probs(
                dim_actions=dim_actions
            )

            return dict(
                action_probs=action_probs,
                mcts_root=copy.deepcopy(mcts_root),
            )

    return planner


@link
@dataclass
class PlannerLaw:
    mcts: MCTSAsync

    async def __call__(self, is_last, policy_feed):
        if not is_last:
            mcts_root = await self.mcts.run(policy_feed)
            action_probs = mcts_root.get_children_visit_counts_as_probs(
                dim_actions=self.mcts.dim_actions
            )

            return dict(
                action_probs=action_probs,
                mcts_root=copy.deepcopy(mcts_root),
            )


def sample_action(action_probs, temperature=1.0):
    logits = np.log(action_probs) / temperature
    action_probs = np.exp(logits) / np.sum(np.exp(logits))
    return np.random.choice(np.arange(len(action_probs)), p=action_probs)


@link
class ActionSamplerLaw:
    temperature: float = 1.0

    def __call__(self, action_probs):
        action = sample_action(action_probs, temperature=self.temperature)
        return dict(action=action)


def temp(arr, temperature=1.0):
    log = np.log(arr) / temperature
    return np.exp(log) / np.sum(np.exp(log))
