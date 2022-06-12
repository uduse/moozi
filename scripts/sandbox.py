# %%
from dataclasses import dataclass
from functools import partial
from acme.jax.utils import add_batch_dim, squeeze_batch_dim
import mctx
import jax
import moozi as mz
from jax import transfer_guard, vmap
import jax.numpy as jnp
import numpy as np


def scalar_transform(x, epsilon=1e-3):
    return jnp.sign(x) * (jnp.sqrt(jnp.abs(x) + 1) - 1) + epsilon * x


def inverse_scalar_transform(x, epsilon=1e-3):
    return jnp.sign(x) * (
        jnp.power(
            (
                (jnp.sqrt(1 + 4 * epsilon * (jnp.abs(x) + 1 + epsilon)) - 1)
                / (2 * epsilon)
            ),
            2,
        )
        - 1
    )


# %%
def phi(x):
    support_min, support_max = -10, 10
    support_dim = support_max + 1 - support_min

    x = jnp.clip(x, support_min, support_max)
    lower_val = jnp.floor(x).astype(jnp.int32)
    upper_val = jnp.ceil(x + np.finfo(np.float32).eps).astype(jnp.int32)
    lower_factor = upper_val - x
    upper_factor = 1 - lower_factor
    lower_idx = lower_val - support_min
    upper_idx = upper_val - support_min
    vec = jnp.zeros(support_dim)
    vec = vec.at[lower_idx].set(lower_factor)
    vec = vec.at[upper_idx].set(upper_factor)
    return vec


# # %%
# input = jnp.array([-100, 1, 2, 3, 4, 30]).reshape((-1, 1))
# transformed = scalar_transform(input)
# print(transformed)
# vmap(phi)(transformed)

# %%
model = mz.nn.make_model(
    mz.nn.ResNetArchitecture,
    mz.nn.ResNetSpec(
        obs_rows=10,
        obs_cols=10,
        obs_channels=10,
        repr_rows=10,
        repr_cols=10,
        repr_channels=10,
        dim_action=6,
        repr_tower_blocks=2,
        repr_tower_dim=8,
        pred_tower_blocks=2,
        pred_tower_dim=8,
        dyna_tower_blocks=2,
        dyna_tower_dim=8,
        dyna_state_blocks=2,
    ),
)
# %%
random_key = jax.random.PRNGKey(0)
params, state = model.init_params_and_state(random_key)

# %%
env = mz.make_env("MinAtar:Seaquest-v1")
timestep = env.reset()
obs = jnp.array(timestep.observation, dtype=jnp.float32)


# %%
def recurr_fn(params, state, random_key, action, hidden_state):
    trans_feats = mz.nn.TransitionFeatures(hidden_state, action)
    is_training = False
    nn_output, _ = model.trans_inference(params, state, trans_feats, is_training)
    rnn_output = mctx.RecurrentFnOutput(
        reward=nn_output.reward.squeeze(-1),
        discount=jnp.ones_like(nn_output.reward.squeeze(-1)),
        prior_logits=nn_output.policy_logits,
        value=nn_output.value.squeeze(-1),
    )
    return rnn_output, nn_output.hidden_state


def plan(params, state, random_key):
    root_feats = mz.nn.RootFeatures(
        obs=obs,
        player=jnp.array([0], dtype=jnp.int32),
        root_feats=add_batch_dim(root_feats),
    )
    root = mctx.RootFnOutput(
        prior_logits=nn_output.policy_logits,
        value=nn_output.value.squeeze(-1),
        embedding=nn_output.hidden_state,
    )
    nn_output, state = model.root_inference(
        params, state, root_feats, is_training=False
    )
    policy_output = mctx.muzero_policy(
        params=params,
        rng_key=jax.random.PRNGKey(0),
        root=root,
        recurrent_fn=partial(recurr_fn, state=state),
        num_simulations=10,
    )
    return policy_output


@link
@dataclass
class Planner:
    num_simulations: int
    known_bound_min: Optional[float]
    known_bound_max: Optional[float]
    include_tree: bool = False
    dirichlet_alpha: float = 0.2
    frac: float = 0.2

    async def __call__(
        self,
        is_last,
        policy_feed: PolicyFeed,
        root_inf_fn,
        trans_inf_fn,
    ):
        legal_actions_mask = policy_feed.legal_actions_mask
        if not is_last:
            mcts = MCTSAsync(
                root_inf_fn=root_inf_fn,
                trans_inf_fn=trans_inf_fn,
                dim_action=legal_actions_mask.size,
                num_simulations=self.num_simulations,
                known_bound_min=self.known_bound_min,
                known_bound_max=self.known_bound_max,
                dirichlet_alpha=self.dirichlet_alpha,
                frac=self.frac,
            )
            mcts_root = await mcts.run(policy_feed)
            action_probs = mcts.get_children_visit_counts_as_probs(
                mcts_root,
            )

            update = dict(
                action_probs=action_probs,
                root_value=mcts_root.value,
            )

            if self.include_tree:
                update["mcts_root"] = copy.deepcopy(mcts_root)
        else:
            action_probs = np.ones_like(legal_actions_mask) / legal_actions_mask.size
            update = dict(action_probs=action_probs, root_value=0.0)

        return update