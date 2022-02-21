import collections
from dataclasses import dataclass, field
from typing import Any, Deque, List, Optional, Union

import dm_env
from jax._src.numpy.lax_numpy import isin
import numpy as np
from absl import logging
from acme.utils.tree_utils import stack_sequence_fields
from moozi.core import link, PolicyFeed
from moozi.core.types import BASE_PLAYER
from moozi.replay import StepSample


@link
@dataclass
class FrameStacker:
    num_frames: int = 1
    player: int = 0

    padding: Optional[np.ndarray] = None
    deque: Deque = field(init=False)

    def __post_init__(self):
        self.deque = collections.deque(maxlen=self.num_frames)

    def __call__(self, obs: Union[np.ndarray, List[np.ndarray]], is_last) -> Any:
        if is_last:
            self.reset()
        else:
            self.add(obs)
            return self.get()

    def add(self, obs: Union[np.ndarray, List[np.ndarray]]):
        assert isinstance(obs, (np.ndarray, list))
        player_obs = self._get_player_obs(obs)

        if self.padding is None:
            self.padding = np.zeros_like(player_obs)

        self.deque.append(player_obs)

    def reset(self):
        self.deque.clear()

    def get(self):
        stacked_frames = np.concatenate(list(self.deque), axis=-1)
        num_frames_to_pad = self.num_frames - len(self.deque)
        if num_frames_to_pad > 0:
            paddings = self.padding.repeat(num_frames_to_pad, axis=-1)
            stacked_frames = np.concatenate([paddings, stacked_frames], axis=-1)
        return stacked_frames

    def _get_player_obs(self, obs: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        if isinstance(obs, np.ndarray):
            return obs
        elif isinstance(obs, list):
            return obs[self.player]

# @link
# def stack_frames(obs: np.ndarray, is_last: bool):
#     assert isinstance(obs, np.ndarray)

#     frame_stacker: FrameStackerV2 = stack_frames.scratch

#     if frame_stacker.padding is None:
#         frame_stacker.padding = np.zeros_like(obs)

#     if is_last:
#         frame_stacker.deque.clear()

#     frame_stacker.deque.append(obs)

#     return dict(
#         stacked_frames=get_stacked_frames(
#             frame_stacker.deque, frame_stacker.num_frames, frame_stacker.padding
#         )
#     )


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

        try:
            to_play = self.env_state.current_player
        except AttributeError:
            to_play = 0

        if 0 <= to_play < self.num_players:
            legal_actions = self._get_legal_actions(timestep)
            legal_actions_curr_player = legal_actions[to_play]
        else:
            legal_actions_curr_player = None

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
                timestep.observation[i].legal_actions
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
            return timestep.reward[BASE_PLAYER]


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
        to_play,
        action,
        reward,
        root_value,
        is_first,
        is_last,
        action_probs,
        output_buffer,
    ):
        if isinstance(obs, list):
            # assume perfect information (same obs for both players)
            obs = obs[BASE_PLAYER]

        step_record = StepSample(
            frame=obs,
            last_reward=reward,
            is_first=is_first,
            is_last=is_last,
            to_play=to_play,
            root_value=root_value,
            action_probs=action_probs,
            action=action,
        ).cast()

        self.traj_buffer.append(step_record)

        if is_last:
            traj = stack_sequence_fields(self.traj_buffer)
            self.traj_buffer.clear()
            return dict(output_buffer=output_buffer + (traj,))


@link
def set_policy_feed(is_last, stacked_frames, legal_actions_mask, to_play):
    if not is_last:
        feed = PolicyFeed(
            stacked_frames=stacked_frames,
            to_play=to_play,
            legal_actions_mask=legal_actions_mask,
            random_key=None,
        )
        return dict(policy_feed=feed)
