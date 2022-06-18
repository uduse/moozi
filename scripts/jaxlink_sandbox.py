# %%
import tree
import chex
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


class BatchedTrajWriter:
    def __init__(self, batch_size: int) -> None:
        self._step_samples: List[List[StepSample]] = [[] for _ in range(batch_size)]

    def __call__(
        self,
        obs,
        to_play,
        action,
        reward,
        root_value,
        is_first,
        is_last,
        action_probs,
        legal_actions_mask,
        output_buffer,
    ):
        for i in range(len(self._step_samples)):
            step_sample = StepSample(
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

            self._step_samples[i].append(step_sample)
            if is_last[i]:
                traj = stack_sequence_fields(self._step_samples[i])
                self._step_samples[i].clear()
                output_buffer = output_buffer + (traj,)

        return dict(output_buffer=output_buffer)


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


def make_planner(
    model: mz.nn.NNModel,
    num_simulations: int = 10,
    dirichlet_fraction: float = 0.25,
    dirichlet_alpha: float = 0.3,
    temperature: float = 1.0,
):
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
            num_simulations=num_simulations,
            dirichlet_fraction=dirichlet_fraction,
            dirichlet_alpha=dirichlet_alpha,
            temperature=temperature,
        )
        stats = policy_output.search_tree.summary()
        return {
            "action_probs": policy_output.action_weights,
            "root_value": stats.value,
            "random_key": random_key,
        }

    return planner


# %%
class Universe:
    def __init__(self, tape, vec_env, agent) -> None:
        self._tape = tape
        self._vec_env = vec_env
        self._agent = agent

        self._env_timer = mz.utils.WallTimer()
        self._agent_timer = mz.utils.WallTimer()

    def tick(self):
        self._env_timer.start()
        self._tape = self._vec_env(self._tape)
        self._env_timer.end()

        self._agent_timer.start()
        self._tape = self._agent(self._tape)
        self._agent_timer.end()

    @property
    def tape(self):
        return self._tape


class UniverseInterrupt(Exception):
    pass


# %%
def slice_tape(tape, exclude: Set[str]):
    return {k: v for k, v in tape.items() if k not in exclude}


# %%
class Agent:
    def __init__(self, model: mz.nn.NNModel, num_envs: int):
        frame_stacker = link(stack_frames)
        planner = link(make_planner(model))
        traj_writer = link_class(BatchedTrajWriter)(num_envs)

        @partial(jax.jit, backend="gpu")
        @chex.assert_max_traces(n=1)
        def policy(tape):
            tape = frame_stacker(tape)
            tape = planner(tape)
            return tape

        def run(tape):
            tape_slice = slice_tape(tape, {"output_buffer", "signals"})
            tape_slice = policy(tape_slice)
            tape.update(tape_slice)

            tape = traj_writer(tape)
            return tape

        self._run = run

    def __call__(self, tape):
        return self._run(tape)


# %%
num_envs = 2
num_stacked_frames = 1
num_actions = 3

model = mz.nn.make_model(
    mz.nn.MLPArchitecture,
    mz.nn.MLPSpec(
        obs_rows=6,
        obs_cols=6,
        obs_channels=num_stacked_frames,
        repr_rows=6,
        repr_cols=6,
        repr_channels=1,
        dim_action=3,
    ),
)
random_key = jax.random.PRNGKey(0)
random_key, new_key = jax.random.split(random_key)
params, state = model.init_params_and_state(new_key)

# %%
random_key, new_key = jax.random.split(random_key)


def make_tape(seed, num_envs, num_actions, num_stacked_frames):
    tape = {}
    tape["root_value"] = np.zeros(num_envs, dtype=np.float32)
    tape["obs"] = np.zeros((num_envs, 6, 6, 1), dtype=np.float32)
    tape["is_first"] = np.full(num_envs, fill_value=False, dtype=bool)
    tape["is_last"] = np.full(num_envs, fill_value=True, dtype=bool)
    tape["action"] = np.full(num_envs, fill_value=0, dtype=np.int32)
    tape["action_probs"] = np.full(
        (num_envs, num_actions), fill_value=0, dtype=np.float32
    )
    tape["stacked_frames"] = np.zeros(
        (num_envs, 6, 6, num_stacked_frames), dtype=np.float32
    )
    tape["random_key"] = new_key
    tape["params"] = params
    tape["state"] = state
    tape["output_buffer"] = tuple()
    return tape


# %%
tape = make_tape(0, num_envs, num_actions, num_stacked_frames)
vec_env = link_class(OpenSpielVecEnv)(make_env_law, num_envs)
agent = Agent(model, num_envs)
universe = Universe(tape, vec_env, agent)
universe.tick()

# %%
num_ticks = 50
timer = mz.utils.WallTimer()
timer.start()
# jax.profiler.start_trace("/tmp/tensorboard")
for i in range(num_ticks):
    print(f"{i=}")
    universe.tick()
timer.end()
# jax.profiler.stop_trace()

num_interactions = num_envs * num_ticks
interactions_per_second = num_interactions / timer.delta

# %%
print(f"{interactions_per_second=}")

# %%
print(f"{universe._agent_timer.delta=}")
print(f"{universe._env_timer.delta=}")