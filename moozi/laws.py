import collections
from dataclasses import dataclass, field
from typing import Any, Deque, List, Optional, Union

import dm_env
from jax._src.numpy.lax_numpy import isin
import numpy as np
from absl import logging
from acme.utils.tree_utils import stack_sequence_fields
from moozi.core import link, PolicyFeed
from moozi.replay import StepSample


@dataclass
class _FrameStacker:
    num_frames: int = 1

    padding: Optional[np.ndarray] = None
    deque: Deque = field(init=False)

    def __post_init__(self):
        self.deque = collections.deque(maxlen=self.num_frames)

    def __call__(self, obs: np.ndarray, is_last) -> Any:
        assert isinstance(obs, np.ndarray)
        if self.padding is None:
            self.padding = np.zeros_like(obs)

        if is_last:
            self.deque.clear()

        self.deque.append(obs)

        return dict(stacked_frames=self._get_stacked_frames())

    def _get_stacked_frames(self):
        stacked_frames = np.array(list(self.deque))
        num_frames_to_pad = self.num_frames - len(self.deque)
        if num_frames_to_pad > 0:
            paddings = np.stack(
                [np.copy(self.padding) for _ in range(num_frames_to_pad)], axis=0
            )
            stacked_frames = np.append(paddings, np.array(list(self.deque)), axis=0)
        return stacked_frames


FrameStacker = link(_FrameStacker)


@link
@dataclass
class PlayerFrameStacker(_FrameStacker):
    player: int = 0

    def __call__(self, obs: List[np.ndarray], is_last) -> Any:
        assert isinstance(obs, list)
        return super().__call__(obs[self.player], is_last)


@link
def update_episode_stats(
    is_last, reward, sum_episodic_reward, num_episodes, universe_id
):
    if is_last:
        sum_episodic_reward = sum_episodic_reward + reward
        num_episodes = num_episodes + 1
        avg_episodic_reward = round(sum_episodic_reward / num_episodes, 3)

        result = dict(
            num_episodes=num_episodes,
            sum_episodic_reward=sum_episodic_reward,
            avg_episodic_reward=avg_episodic_reward,
        )
        logging.debug({**dict(universe_id=universe_id), **result})

        return result


@link
def output_last_step_reward(is_last, reward, output_buffer):
    if is_last:
        output_buffer = output_buffer + (reward,)
        return dict(output_buffer=output_buffer)


@link
@dataclass
class EnvironmentLaw:
    env_state: dm_env.Environment
    num_players: int = 1

    def __call__(self, obs, is_last, action: int):
        if obs is None or is_last:
            timestep = self.env_state.reset()
        else:
            timestep = self.env_state.step([action])

        return dict(
            obs=self._get_observation(timestep),
            is_first=timestep.first(),
            is_last=timestep.last(),
            to_play=self.env_state.current_player,
            reward=self._get_reward(timestep, self.num_players),
            legal_actions_mask=self._get_legal_actions(timestep),
        )

    @staticmethod
    def _get_observation(timestep: dm_env.TimeStep):
        if isinstance(timestep.observation, list):
            return [
                timestep.observation[i].observation
                for i in range(len(timestep.observation))
            ]
        else:
            raise NotImplementedError

    @staticmethod
    def _get_legal_actions(timestep: dm_env.TimeStep):
        if isinstance(timestep.observation, list):
            return [
                timestep.observation[i].legal_actions_mask
                for i in range(len(timestep.observation))
            ]
        else:
            raise NotImplementedError

    @staticmethod
    def _get_reward(timestep: dm_env.TimeStep, num_players: int):
        if timestep.reward is None:
            return [0.0] * num_players
        elif isinstance(timestep.reward, np.ndarray):
            assert len(timestep.reward) == num_players
            return timestep.reward


@link
def increment_tick(num_ticks):
    return {"num_ticks": num_ticks + 1}


@link
def set_random_action_from_timestep(is_last, legal_actions):
    action = -1
    if not is_last:
        random_action = np.random.choice(np.flatnonzero(legal_actions == 1))
        action = random_action
    return dict(action=action)


@link
@dataclass
class TrajectoryOutputWriter:
    traj_buffer: list = field(default_factory=list)

    def __call__(
        self,
        obs,
        action,
        reward,
        root_value,
        is_first,
        is_last,
        action_probs,
        output_buffer,
    ):
        step_record = StepSample(
            frame=obs,
            reward=reward,
            is_first=is_first,
            is_last=is_last,
            action=action,
            root_value=root_value,
            action_probs=action_probs,
        ).cast()

        self.traj_buffer.append(step_record)

        if is_last:
            traj = stack_sequence_fields(self.traj_buffer)
            self.traj_buffer.clear()
            return dict(output_buffer=output_buffer + (traj,))


@link
def set_policy_feed(is_last, stacked_frames, legal_actions_mask):
    if not is_last:
        feed = PolicyFeed(
            features=stacked_frames,
            legal_actions_mask=legal_actions_mask,
            random_key=None,
        )
        return dict(policy_feed=feed)
