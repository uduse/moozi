# %%
import functools
import inspect
import pickle
from dataclasses import asdict, dataclass
from typing import Callable, Optional

import dm_env
import gym
import jax
import jax.numpy as jnp
import moozi as mz
import numpy as np
from acme.utils.tree_utils import stack_sequence_fields, unstack_sequence_fields
from moozi import Tape
from moozi.core.types import PolicyFeed


def link(fn):
    keys = inspect.signature(fn).parameters.keys()

    if inspect.isclass(fn):
        fn = fn.__call__

    def _wrapper(d):
        kwargs = {}
        for k in keys:
            kwargs[k] = d[k]
        updates = fn(**kwargs)
        d.update(updates)
        return d

    return _wrapper


def link_class(cls):
    @dataclass
    class _LinkClassWrapper:
        class_: type

        def __call__(self, *args, **kwargs):
            return link(self.class_(*args, **kwargs))

    return _LinkClassWrapper(cls)


class OpenSpielVecEnv:
    def __init__(self, env_factory: Callable, num_envs: int):
        self._envs = [env_factory() for _ in range(num_envs)]

    def __call__(self, is_last, action):
        updates_list = []
        for law, is_last_, action_ in zip(self._envs, is_last, action):
            updates = law(is_last=is_last_, action=action_)
            updates_list.append(updates)
        print(updates_list[0]["obs"][0].shape)
        return stack_sequence_fields(updates_list)


# %%
@dataclass
class OpenSpielEnv:
    env: dm_env.Environment
    num_players: int = 1

    _legal_actions_mask_padding: Optional[np.ndarray] = None

    def __call__(self, is_last, action: int):
        if is_last.item():
            timestep = self.env.reset()
        else:
            timestep = self.env.step([action])

        try:
            to_play = self.env.current_player
        except AttributeError:
            to_play = 0

        if 0 <= to_play < self.num_players:
            legal_actions = self._get_legal_actions(timestep)
            legal_actions_curr_player = legal_actions[to_play]
            if legal_actions_curr_player is None:
                assert self._legal_actions_mask_padding is not None
                legal_actions_curr_player = self._legal_actions_mask_padding.copy()
        else:
            assert self._legal_actions_mask_padding is not None
            legal_actions_curr_player = self._legal_actions_mask_padding.copy()

        should_init_padding = (
            self._legal_actions_mask_padding is None
            and legal_actions_curr_player is not None
        )
        if should_init_padding:
            self._legal_actions_mask_padding = np.ones_like(legal_actions_curr_player)

        return dict(
            obs=self._get_observation(timestep),
            is_first=timestep.first(),
            is_last=timestep.last(),
            to_play=to_play,
            reward=self._get_reward(timestep, self.num_players),
            legal_actions_mask=legal_actions_curr_player,
        )

    @staticmethod
    def _get_observation(timestep: dm_env.TimeStep):
        if isinstance(timestep.observation, list):
            return timestep.observation[0].observation
        else:
            raise NotImplementedError

    @staticmethod
    def _get_legal_actions(timestep: dm_env.TimeStep):
        if isinstance(timestep.observation, list):
            return timestep.observation[0].legal_actions
        else:
            raise NotImplementedError

    @staticmethod
    def _get_reward(timestep: dm_env.TimeStep, num_players: int):
        if timestep.reward is None:
            return 0.0
        elif isinstance(timestep.reward, np.ndarray):
            assert len(timestep.reward) == num_players
            return timestep.reward[mz.BASE_PLAYER]


def make_env():
    return mz.make_env("OpenSpiel:catch(rows=6,columns=6)")


def make_env_law():
    return OpenSpielEnv(make_env())


# %%
num_envs = 2
vec_env = link_class(OpenSpielVecEnv)(make_env_law, num_envs)

# %%
tape = asdict(Tape())
tape["obs"] = np.zeros((num_envs, 6, 6, 1), dtype=np.float32)
tape["is_first"] = np.full((num_envs), fill_value=True, dtype=bool)
tape["is_last"] = np.full((num_envs), fill_value=False, dtype=bool)
tape["action"] = np.full((num_envs), fill_value=0, dtype=np.int32)

# %%
vec_env(tape)


@link
def make_policy_feed(stacked_frames, legal_actions_mask, to_play):
    feed = PolicyFeed(
        stacked_frames=stacked_frames,
        to_play=to_play,
        legal_actions_mask=legal_actions_mask,
        random_key=jax.random.PRNGKey(0),
    )
    return dict(policy_feed=feed)


class Universe:
    def __init__(self, vec_env) -> None:
        self._vec_env = vec_env
        self._vec_agent = None

    def tick(self, tape):
        self._vec_env(tape)
        self._vec_agent(tape)
        return tape


# %%
def stack_frame(stacked_frames, obs):
    shifted = jnp.roll(stacked_frames, shift=1, axis=0)
    filed = stacked_frames.at[]


num_stacked_frames = 3
stacked_frames = np.zeros((num_envs, 6, 6, num_stacked_frames), dtype=np.float32)
tape = {"stacked_frames": stacked_frames}


# %%
# model = mz.nn.make_model(
#     mz.nn.ResNetArchitecture,
#     mz.nn.ResNetSpec(
#         obs_rows=10,
#         obs_cols=10,
#         obs_channels=10,
#         repr_rows=10,
#         repr_cols=10,
#         repr_channels=10,
#         dim_action=6,
#         repr_tower_blocks=2,
#         repr_tower_dim=8,
#         pred_tower_blocks=2,
#         pred_tower_dim=8,
#         dyna_tower_blocks=2,
#         dyna_tower_dim=8,
#         dyna_state_blocks=2,
#     ),
# )
# # %%
# random_key = jax.random.PRNGKey(0)
# params, state = model.init_params_and_state(random_key)

# # %%
# env = mz.make_env("MinAtar:Seaquest-v1")
# timestep = env.reset()
# obs = jnp.array(timestep.observation, dtype=jnp.float32)


# # %%
# def recurr_fn(params, state, random_key, action, hidden_state):
#     trans_feats = mz.nn.TransitionFeatures(hidden_state, action)
#     is_training = False
#     nn_output, _ = model.trans_inference(params, state, trans_feats, is_training)
#     rnn_output = mctx.RecurrentFnOutput(
#         reward=nn_output.reward.squeeze(-1),
#         discount=jnp.ones_like(nn_output.reward.squeeze(-1)),
#         prior_logits=nn_output.policy_logits,
#         value=nn_output.value.squeeze(-1),
#     )
#     return rnn_output, nn_output.hidden_state


# def plan(params, state, random_key):
#     root_feats = mz.nn.RootFeatures(
#         obs=obs,
#         player=jnp.array([0], dtype=jnp.int32),
#         root_feats=add_batch_dim(root_feats),
#     )
#     root = mctx.RootFnOutput(
#         prior_logits=nn_output.policy_logits,
#         value=nn_output.value.squeeze(-1),
#         embedding=nn_output.hidden_state,
#     )
#     nn_output, state = model.root_inference(
#         params, state, root_feats, is_training=False
#     )
#     policy_output = mctx.muzero_policy(
#         params=params,
#         rng_key=jax.random.PRNGKey(0),
#         root=root,
#         recurrent_fn=partial(recurr_fn, state=state),
#         num_simulations=10,
#     )
#     return policy_output


# @link
# @dataclass
# class Planner:
#     num_simulations: int
#     known_bound_min: Optional[float]
#     known_bound_max: Optional[float]
#     include_tree: bool = False
#     dirichlet_alpha: float = 0.2
#     frac: float = 0.2

#     async def __call__(
#         self,
#         is_last,
#         policy_feed: PolicyFeed,
#         root_inf_fn,
#         trans_inf_fn,
#     ):
#         legal_actions_mask = policy_feed.legal_actions_mask
#         if not is_last:
#             mcts = MCTSAsync(
#                 root_inf_fn=root_inf_fn,
#                 trans_inf_fn=trans_inf_fn,
#                 dim_action=legal_actions_mask.size,
#                 num_simulations=self.num_simulations,
#                 known_bound_min=self.known_bound_min,
#                 known_bound_max=self.known_bound_max,
#                 dirichlet_alpha=self.dirichlet_alpha,
#                 frac=self.frac,
#             )
#             mcts_root = await mcts.run(policy_feed)
#             action_probs = mcts.get_children_visit_counts_as_probs(
#                 mcts_root,
#             )

#             update = dict(
#                 action_probs=action_probs,
#                 root_value=mcts_root.value,
#             )

#             if self.include_tree:
#                 update["mcts_root"] = copy.deepcopy(mcts_root)
#         else:
#             action_probs = np.ones_like(legal_actions_mask) / legal_actions_mask.size
#             update = dict(action_probs=action_probs, root_value=0.0)

#         return update
# %%

# %%
