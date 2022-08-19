from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pyspiel
from flax import struct
from moozi.core import StepSample
from moozi.core.env import GIIEnvOut, GIIVecEnv
from moozi.core.utils import (
    HistoryStacker,
)
from moozi.nn import NNModel, RootFeatures
from moozi.planner import Planner


class PolicyFeed(struct.PyTreeNode):
    stacker_state: HistoryStacker.StackerState
    params: hk.Params
    state: hk.State
    env_out: GIIEnvOut
    last_action: chex.Array
    random_key: chex.PRNGKey

    stacker: HistoryStacker = struct.field(pytree_node=False)
    planner: Planner = struct.field(pytree_node=False)


class PolicyOut(struct.PyTreeNode):
    stacker_state: HistoryStacker.StackerState
    planner_out: Planner.PlannerOut


PolicyType = Callable[[PolicyFeed], PolicyOut]


def policy(
    policy_feed: PolicyFeed,
) -> PolicyOut:
    stacker_state = jax.vmap(policy_feed.stacker.apply)(
        state=policy_feed.stacker_state,
        frame=policy_feed.env_out.frame,
        action=policy_feed.last_action,
        is_first=policy_feed.env_out.is_first,
    )
    root_feats = RootFeatures(
        frames=stacker_state.frames,
        actions=stacker_state.actions,
        to_play=policy_feed.env_out.to_play,
    )
    planner_feed = Planner.PlannerFeed(
        params=policy_feed.params,
        state=policy_feed.state,
        root_feats=root_feats,
        legal_actions=policy_feed.env_out.legal_actions,
        random_key=policy_feed.random_key,
    )
    planner_out = policy_feed.planner.run(planner_feed)
    return PolicyOut(stacker_state=stacker_state, planner_out=planner_out)


class GII:
    def __init__(
        self,
        env_name: str,
        stacker: HistoryStacker,
        planner: Union[Planner, Dict[int, Planner]],
        params: Union[hk.Params, Dict[int, hk.Params]],
        state: Union[hk.State, Dict[int, hk.State]],
        random_key: chex.PRNGKey,
        num_envs: int = 1,
    ):
        self.env = GIIVecEnv(env_name, num_envs=num_envs)
        self.env_feed = self.env.init()
        self.env_out: Optional[GIIEnvOut] = None
        self.planner_out: Optional[Planner.PlannerOut] = None

        self.stacker = stacker
        self.stacker_state = jax.vmap(stacker.init, axis_size=num_envs)()

        self.random_key = random_key
        self.planner = planner
        self.params = params
        self.state = state

        self.policy: PolicyType = jax.jit(policy, backend="gpu")

    @staticmethod
    def _select_for_player(data: Union[Any, Dict[int, Any]], to_play: int):
        if to_play == pyspiel.PlayerId.TERMINAL:
            to_play = 0
        if isinstance(data, dict) and (to_play in data):
            return data[to_play]
        else:
            return data

    def _select_planner(self, to_play: int) -> Planner:
        return self._select_for_player(self.planner, to_play)

    def _select_params_and_state(self, to_play: int) -> Tuple[hk.Params, hk.State]:
        params = self._select_for_player(self.params, to_play)
        state = self._select_for_player(self.state, to_play)
        return params, state

    def _next_key(self):
        self.random_key, next_key = jax.random.split(self.random_key)
        return next_key

    def tick(self) -> StepSample:
        env_out = self.env.step(self.env_feed)
        to_play = int(env_out.to_play)

        params, state = self._select_params_and_state(to_play)
        planner = self._select_planner(to_play)

        policy_feed = PolicyFeed(
            params=params,
            state=state,
            planner=planner,
            env_out=env_out,
            last_action=self.env_feed.action,
            stacker=self.stacker,
            stacker_state=self.stacker_state,
            random_key=self._next_key(),
        )
        policy_out = self.policy(policy_feed)
        action = policy_out.planner_out.action

        self.stacker_state = policy_out.stacker_state
        self.env_feed.reset = env_out.is_last
        self.env_feed.action = np.array(action)
        self.env_out = env_out
        self.planner_out = policy_out.planner_out

        return StepSample(
            frame=env_out.frame,
            last_reward=env_out.reward,
            is_first=env_out.is_first,
            is_last=env_out.is_last,
            to_play=env_out.to_play,
            legal_actions_mask=env_out.legal_actions,
            root_value=policy_out.planner_out.root_value,
            action_probs=policy_out.planner_out.action_probs,
            action=action,
        )
