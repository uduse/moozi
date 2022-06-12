# %%
import mctx
import functools
import inspect
import os
import pickle
from dataclasses import asdict, dataclass, field
from functools import partial
from re import I
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from acme.jax.utils import add_batch_dim

import dm_env
import gym
import haiku as hk
import jax
import jax.numpy as jnp
import moozi as mz
import numpy as np
import ray
import tree
from absl import logging
from acme.utils.tree_utils import stack_sequence_fields, unstack_sequence_fields
from loguru import logger
from moozi.core.types import PolicyFeed, StepSample
from moozi.nn import RootFeatures, TransitionFeatures


def link(fn):
    keys = inspect.signature(fn).parameters.keys()

    if inspect.isclass(fn):
        fn = fn.__call__

    def _wrapper(d):
        kwargs = {}
        for k in keys:
            kwargs[k] = d[k]
        updates = fn(**kwargs)
        d = d.copy()
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
        for env, is_last_, action_ in zip(self._envs, is_last, action):
            updates = env(is_last=is_last_, action=action_)
            updates_list.append(updates)
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


def make_policy_feed(stacked_frames, legal_actions_mask, to_play, random_key):
    random_key, new_key = jax.random.split(random_key)
    feed = PolicyFeed(
        stacked_frames=stacked_frames,
        to_play=to_play,
        legal_actions_mask=legal_actions_mask,
        random_key=new_key,
    )
    return dict(policy_feed=feed, random_key=random_key)


@link
def write_traj_batched(
    obs,
    to_play,
    action,
    reward,
    root_value,
    is_first,
    is_last,
    action_probs,
    legal_actions_mask,
    step_records: Tuple[Tuple[StepSample, ...], ...],
    output_buffer,
):
    new_step_records = list(step_records)
    for i in range(len(step_records)):
        step_record = StepSample(
            frame=obs[i],
            last_reward=reward[i],
            is_first=is_first[i],
            is_last=is_last[i],
            to_play=to_play[i],
            legal_actions_mask=legal_actions_mask[i],
            root_value=root_value[i],
            action_probs=action_probs[i],
            action=action[i],
            weight=1.0,
        )

        if is_last[i]:
            traj = stack_sequence_fields(step_records[i] + (step_record,))
            output_buffer = output_buffer + (traj,)
            new_step_records[i] = tuple()
        else:
            new_step_records[i] = step_records[i] + (step_record,)

    return dict(step_records=tuple(new_step_records), output_buffer=output_buffer)


# %%
def stack_frames(stacked_frames, obs):
    ret = jnp.append(stacked_frames, obs, axis=-1)
    ret = ret[..., np.array(obs.shape[-1]) :]
    return {"stacked_frames": ret}


# %%
@dataclass(repr=False)
class RolloutWorkerVec:
    name: str = "rollout_worker_vec"
    universe: "Universe" = field(init=False)

    def run(self, termination: str = "episode"):
        try:
            while 1:
                self.universe.tick()
        except UniverseInterrupt:
            pass
        finally:
            return

    def set_params_and_state(
        self, params_and_state: Union[ray.ObjectRef, Tuple[hk.Params, hk.State]]
    ):
        if isinstance(params_and_state, ray.ObjectRef):
            params, state = ray.get(params_and_state)
        else:
            params, state = params_and_state
        self.universe.tape["params"] = params
        self.universe.tape["state"] = state


# %%
def make_paritial_recurr_fn(state):
    def recurr_fn(params, random_key, action, hidden_state):
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

    return recurr_fn


def make_planner(model: mz.nn.NNModel):
    def planner(params: hk.Params, state: hk.State, stacked_frames, random_key):
        is_training = False
        random_key, new_key = jax.random.split(random_key, 2)
        root_feats = mz.nn.RootFeatures(
            obs=stacked_frames,
            player=np.zeros((stacked_frames.shape[0]), dtype=np.int32),
        )
        nn_output, _ = model.root_inference(params, state, root_feats, is_training)
        root = mctx.RootFnOutput(
            prior_logits=nn_output.policy_logits,
            value=nn_output.value.squeeze(-1),
            embedding=nn_output.hidden_state,
        )
        nn_output, _ = model.root_inference(params, state, root_feats, is_training)
        policy_output = mctx.muzero_policy(
            params=params,
            rng_key=new_key,
            root=root,
            recurrent_fn=make_paritial_recurr_fn(state),
            num_simulations=20,
        )
        stats = policy_output.search_tree.summary()
        return {
            "action_probs": policy_output.action_weights,
            "root_value": stats.value,
        }

    return planner


# %%
class Universe:
    def __init__(self, tape, vec_env, agent) -> None:
        self._tape = tape
        self._vec_env = vec_env
        self._agent = agent

    def tick(self):
        self._tape = self._vec_env(self._tape)
        self._tape = self._agent(self._tape)
        return self._tape


class UniverseInterrupt(Exception):
    pass


@dataclass
class Tape:
    # statistics
    num_ticks: int = 0
    num_episodes: int = 0
    avg_episodic_reward: float = 0
    sum_episodic_reward: float = 0

    # environment
    obs: np.ndarray = None
    is_first: bool = True
    is_last: bool = False
    to_play: int = 0
    reward: float = 0.0
    action: int = 0
    discount: float = 1.0
    legal_actions_mask: np.ndarray = np.array(1)

    # planner output
    root_value: float = 0
    action_probs: np.ndarray = np.array(0.0)
    # mcts_root: Optional[Any] = None

    # nn
    params: hk.Params = None
    state: hk.State = None

    # player inputs
    stacked_frames: np.ndarray = np.array(0)
    policy_feed: Optional[PolicyFeed] = None

    input_buffer: tuple = field(default_factory=tuple)
    output_buffer: tuple = field(default_factory=tuple)

    signals: Dict[str, bool] = field(default_factory=lambda: {"exit": False})


# %%
def slice_tape(tape, exclude: Set[str]):
    return {k: v for k, v in tape.items() if k not in exclude}


# %%
class Agent:
    def __init__(self, model: mz.nn.NNModel):
        frame_stacker = link(stack_frames)
        policy_feed_maker = link(make_policy_feed)
        planner = link(make_planner(model))
        traj_writer = link(write_traj_batched)

        @jax.jit
        def policy(tape):
            tape = frame_stacker(tape)
            tape = policy_feed_maker(tape)
            tape = planner(tape)
            return tape

        def run(tape):
            tape = policy(tape)
            tape = traj_writer(tape)
            return tape

        self._run = run

    def __call__(self, tape):
        return self._run(tape)


# %%
num_envs = 2
num_stacked_frames = 2

model = mz.nn.make_model(
    mz.nn.MLPArchitecture,
    mz.nn.MLPSpec(
        obs_rows=6,
        obs_cols=6,
        obs_channels=num_stacked_frames,
        repr_rows=6,
        repr_cols=6,
        repr_channels=4,
        dim_action=3,
    ),
)
random_key = jax.random.PRNGKey(0)
random_key, new_key = jax.random.split(random_key)
params, state = model.init_params_and_state(new_key)

# %%
random_key, new_key = jax.random.split(random_key)
tape = asdict(Tape())
tape["obs"] = np.zeros((num_envs, 6, 6, 1), dtype=np.float32)
tape["is_first"] = np.full((num_envs), fill_value=False, dtype=bool)
tape["is_last"] = np.full((num_envs), fill_value=True, dtype=bool)
tape["action"] = np.full((num_envs), fill_value=0, dtype=np.int32)
tape["stacked_frames"] = np.zeros(
    (num_envs, 6, 6, num_stacked_frames), dtype=np.float32
)
tape["random_key"] = new_key
tape["params"] = params
tape["state"] = state

# %%
vec_env = link_class(OpenSpielVecEnv)(make_env_law, num_envs)
agent = Agent(model)
universe = Universe(tape, vec_env, agent)

# %%
for i in range(6):
    print(f"{i=}")
    tape = universe.tick()
    print()
