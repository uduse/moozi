# %%
import jax
from IPython import display
import numpy as np
from moozi.core.types import PolicyFeed
from moozi.nn import (
    RootInferenceFeatures,
    NNOutput,
    NeuralNetwork,
    TransitionInferenceFeatures,
)
from moozi.policy.mcts_async import MCTSAsync
from moozi.policy.mcts_core import (
    Node,
    anytree_to_png,
    anytree_to_text,
    convert_to_anytree,
    anytree_display_in_notebook,
    anytree_filter_node,
    get_next_player,
    SearchStrategy,
)


# %%
def init_inf_fn(features: RootInferenceFeatures):
    print("features:", features)
    value = np.round(np.random.rand() * 2 - 1, 1)
    output = NNOutput(
        value=value,
        reward=np.array(0.0),
        policy_logits=np.array([0.5, 0.5]),
        hidden_state=np.array(features.player),
    )
    print(output)
    return output


def recurr_inf_fn(features: TransitionInferenceFeatures):
    print("features:", features)
    next_player = get_next_player(SearchStrategy.TWO_PLAYER, int(features.hidden_state))
    value = np.round(np.random.rand() * 2 - 1, 1)
    reward = np.round(np.random.rand() * 2 - 1, 1)
    if features.action == 0:
        value -= 0.2

    # if next_player == 0:
    #     reward = -reward
    # else:
    #     value = -value
    output = NNOutput(
        value=value,
        reward=reward,
        policy_logits=np.array([0.5, 0.5]),
        hidden_state=np.array(next_player),
    )
    print(output)
    return output


# %%
mcts = MCTSAsync(
    dim_action=2,
    num_simulations=1,
    init_inf_fn=init_inf_fn,
    recurr_inf_fn=recurr_inf_fn,
    discount=1.0,
)

# %%
feed = PolicyFeed(
    stacked_frames=None, to_play=0, legal_actions_mask=np.ones(2), random_key=0
)

# %%
node = await mcts.get_root(feed)
node.last_reward = 0.0

# %%
anytree_node = convert_to_anytree(node)
anytree_filter_node(anytree_node, lambda n: n.visits > 0)
anytree_display_in_notebook(anytree_node)

# %%
await mcts.simulate_once(node)
anytree_node = convert_to_anytree(node)
anytree_filter_node(anytree_node, lambda n: n.visits > 0)
anytree_display_in_notebook(anytree_node)

# %%
for _ in range(100):
    await mcts.simulate_once(node)
anytree_node = convert_to_anytree(node)
anytree_filter_node(anytree_node, lambda n: n.visits > 0)
anytree_display_in_notebook(anytree_node)

# %%
node.select_leaf()