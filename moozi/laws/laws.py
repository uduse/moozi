from dataclasses import dataclass, field
import numpy as np
import collections
from typing import Any, Deque, Optional
from moozi.link import link

from absl import logging
import dm_env


@link
@dataclass
class FrameStacker:
    num_frames: int = 1

    padding: Optional[np.ndarray] = None
    deque: Deque = field(init=False)

    def __post_init__(self):
        self.deque = collections.deque(maxlen=self.num_frames)

    def __call__(self, obs, is_last) -> Any:
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


@link
def update_episode_stats(timestep, sum_episodic_reward, num_episodes, universe_id):
    if timestep.last():
        sum_episodic_reward = sum_episodic_reward + float(timestep.reward)
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

    def __call__(self, obs, is_last, action: int):
        if obs is None or is_last:
            timestep = self.env_state.reset()
        else:
            timestep = self.env_state.step([action])

        if timestep.reward is None:
            reward = 0.0
        else:
            reward = float(np.nan_to_num(timestep.reward))

        return dict(
            obs=self._get_observation(timestep),
            is_first=timestep.first(),
            is_last=timestep.last(),
            to_play=self.env_state.current_player,
            reward=reward,
            legal_actions_mask=self._get_legal_actions(timestep),
        )

    @staticmethod
    def _get_observation(timestep: dm_env.TimeStep):
        if isinstance(timestep.observation, list):
            assert len(timestep.observation) == 1
            return timestep.observation[0].observation
        else:
            raise NotImplementedError

    @staticmethod
    def _get_legal_actions(timestep: dm_env.TimeStep):
        if isinstance(timestep.observation, list):
            assert len(timestep.observation) == 1
            return timestep.observation[0].legal_actions
        else:
            raise NotImplementedError


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
