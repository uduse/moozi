import collections
import jax.numpy as jnp
import uuid
from dataclasses import dataclass, field
from typing import Any, Deque, List, Optional, Union

import dm_env
import numpy as np
from absl import logging
from acme.utils.tree_utils import stack_sequence_fields, unstack_sequence_fields
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from loguru import logger

from moozi.core import BASE_PLAYER, PolicyFeed, StepSample
from moozi.core.link import link
from moozi.core.types import TrainTarget

# TODO: make __call__ not a method but a static method or a function

# @link
# @dataclass
# class FrameStacker:
#     num_frames: int = 1
#     player: int = 0

#     padding: Optional[np.ndarray] = None
#     deque: Deque = field(init=False)

#     def __post_init__(self):
#         self.deque = collections.deque(maxlen=self.num_frames)

#     def __call__(self, obs: Union[np.ndarray, List[np.ndarray]], is_last) -> Any:
#         if is_last:
#             self.reset()
#         else:
#             self.add(obs)
#             return dict(stacked_frames=self.get())

#     def add(self, obs: Union[np.ndarray, List[np.ndarray]]):
#         assert isinstance(obs, (np.ndarray, list))
#         player_obs = self._get_player_obs(obs)

#         if self.padding is None:
#             self.padding = np.zeros_like(player_obs)

#         self.deque.append(player_obs)

#     def reset(self):
#         self.deque.clear()

#     def get(self):
#         stacked_frames = np.concatenate(list(self.deque), axis=-1)
#         num_frames_to_pad = self.num_frames - len(self.deque)
#         if num_frames_to_pad > 0:
#             paddings = self.padding.repeat(num_frames_to_pad, axis=-1)
#             stacked_frames = np.concatenate([paddings, stacked_frames], axis=-1)
#         return stacked_frames

#     def _get_player_obs(self, obs: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
#         if isinstance(obs, np.ndarray):
#             return obs
#         elif isinstance(obs, list):
#             return obs[self.player]
#         else:
#             raise ValueError(
#                 f"obs must be np.ndarray or list of np.ndarray, got {type(obs)}"
#             )


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
class OpenSpielEnvLaw:
    env: dm_env.Environment
    num_players: int = 1

    _legal_actions_mask_padding: Optional[np.ndarray] = None

    def __call__(self, obs, is_last, action: int):
        if (obs is None) or is_last:
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
            return 0.0
        elif isinstance(timestep.reward, np.ndarray):
            assert len(timestep.reward) == num_players
            return timestep.reward[BASE_PLAYER]


@link
@dataclass
class AtariEnvLaw:
    env: dm_env.Environment
    record_video: bool = False

    _video_recorder: Optional[VideoRecorder] = None

    def __call__(self, is_first, is_last, action: int):
        if self.record_video:
            if is_first:
                fname = f"/tmp/{str(uuid.uuid4())}.mp4"
                self._video_recorder = VideoRecorder(self.env, fname, enabled=False)
            elif is_last:
                assert self._video_recorder is not None
                self._video_recorder.close()
                logger.info(f"Recorded to {self.env.video_recorder.path}")

        if is_last:
            timestep = self.env.reset()
        else:
            timestep = self.env.step(action)

        if timestep.reward is None:
            reward = 0.0
        else:
            reward = timestep.reward

        legal_actions_mask = np.ones(self.env.action_space.n, dtype=np.float32)

        if self.record_video:
            assert self._video_recorder is not None
            self._video_recorder.capture_frame()

        return dict(
            obs=timestep.observation,
            is_first=timestep.first(),
            is_last=timestep.last(),
            to_play=0,
            reward=reward,
            legal_actions_mask=legal_actions_mask,
        )


@link
@dataclass
class MinAtarEnvLaw:
    env: dm_env.Environment

    def __call__(self, is_first, is_last, action: int):
        if is_last:
            timestep = self.env.reset()
        else:
            timestep = self.env.step(action)

        if timestep.reward is None:
            reward = 0.0
        else:
            reward = timestep.reward

        legal_actions_mask = np.ones(self.env.action_space.n, dtype=np.float32)

        return dict(
            obs=np.array(timestep.observation, dtype=float),
            is_first=timestep.first(),
            is_last=timestep.last(),
            to_play=0,
            reward=reward,
            legal_actions_mask=legal_actions_mask,
        )


@link
@dataclass
class ReanalyzeEnvLaw:
    _curr_traj: Optional[List[StepSample]] = None
    _curr_step: int = 0

    def __call__(self, input_buffer):
        input_buffer_update = input_buffer
        self._curr_step += 1

        if (self._curr_traj is None) or (
            self._curr_step >= self._curr_traj[0].shape[0]
        ):
            self._curr_step = 0
            self._curr_traj = input_buffer[0]
            assert isinstance(self._curr_traj, StepSample)
            input_buffer_update = tuple(input_buffer[1:])

        return dict(
            obs=self._curr_traj.frame[self._curr_step],
            is_first=self._curr_traj.is_first[self._curr_step],
            is_last=self._curr_traj.is_last[self._curr_step],
            to_play=self._curr_traj.to_play[self._curr_step],
            reward=self._curr_traj.last_reward[self._curr_step],
            legal_actions_mask=self._curr_traj.legal_actions_mask[self._curr_step],
            action=self._curr_traj.action[self._curr_step],
            input_buffer=input_buffer_update,
        )


@link
@dataclass
class ReanalyzeEnvLawV2:
    def __call__(self, input_buffer):
        input_buffer_update = tuple(input_buffer[1:])
        train_target: TrainTarget = input_buffer[0]
        assert isinstance(train_target, TrainTarget)
        return dict(
            stacked_frames=train_target.stacked_frames,
            is_last=False,
            legal_actions_mask=np.ones_like(train_target.action_probs),
            input_buffer=input_buffer_update,
        )


def exit_if_no_input(input_buffer):
    if not input_buffer:
        return {"interrupt_exit": True}


# @link
# def increment_tick(num_ticks):
#     return {"num_ticks": num_ticks + 1}


# @link
# def set_random_action_from_timestep(is_last, legal_actions):
#     action = -1
#     if not is_last:
#         random_action = np.random.choice(np.flatnonzero(legal_actions == 1))
#         action = random_action
#     return dict(action=action)


# @link
# @dataclass
# class TrajectoryOutputWriter:
#     traj_buffer: list = field(default_factory=list)

#     def __call__(
#         self,
#         obs,
#         to_play,
#         action,
#         reward,
#         root_value,
#         is_first,
#         is_last,
#         action_probs,
#         legal_actions_mask,
#         output_buffer,
#     ):
#         if isinstance(obs, list):
#             # assume perfect information (same obs for both players)
#             obs = obs[BASE_PLAYER]

#         step_record = StepSample(
#             frame=obs,
#             last_reward=reward,
#             is_first=is_first,
#             is_last=is_last,
#             to_play=to_play,
#             legal_actions_mask=legal_actions_mask,
#             root_value=root_value,
#             action_probs=action_probs,
#             action=action,
#             weight=1.0,
#         ).cast()

#         self.traj_buffer.append(step_record)

#         if is_last:
#             traj = stack_sequence_fields(self.traj_buffer)
#             self.traj_buffer.clear()
#             return dict(output_buffer=output_buffer + (traj,))


# @link
# def make_policy_feed(is_last, stacked_frames, legal_actions_mask, to_play):
#     if not is_last:
#         feed = PolicyFeed(
#             stacked_frames=stacked_frames,
#             to_play=to_play,
#             legal_actions_mask=legal_actions_mask,
#             random_key=None,
#         )
#         return dict(policy_feed=feed)


@dataclass
class TrajWriter:
    num_envs: int

    def malloc(self):
        return {"step_samples": [[] for _ in range(self.num_envs)]}

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
        step_samples,
        output_buffer,
    ):
        for i in range(len(step_samples)):
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

            step_samples[i].append(step_sample)
            if is_last[i]:
                traj = stack_sequence_fields(step_samples[i])
                step_samples[i].clear()
                output_buffer = output_buffer + (traj,)

        return dict(output_buffer=output_buffer)


@dataclass
class BatchFrameStacker:
    num_envs: int
    num_rows: int
    num_cols: int
    num_channels: int
    num_stacked_frames: int

    def malloc(self):
        return {
            "stacked_frames": jnp.zeros(
                (
                    self.num_envs,
                    self.num_rows,
                    self.num_cols,
                    self.num_stacked_frames * self.num_channels,
                ),
                dtype=jnp.float32,
            )
        }

    def __call__(self, stacked_frames, obs):
        ret = jnp.append(stacked_frames, obs, axis=-1)
        ret = ret[..., np.array(obs.shape[-1]) :]
        return {"stacked_frames": ret}

