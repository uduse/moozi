import collections
import inspect
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Union

import dm_env
import jax
import jax.numpy as jnp
import numpy as np
import seaborn as sns
import tree
from absl import logging
from acme.utils.tree_utils import stack_sequence_fields, unstack_sequence_fields
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from loguru import logger
from PIL import Image, ImageDraw, ImageFont

from moozi.core import BASE_PLAYER, PolicyFeed, StepSample
from moozi.core.env import make_env
from moozi.core.link import link, unlink
from moozi.core.tape import include
from moozi.core.types import TrainTarget


# TODO: make __call__ not a method but a static method or a function
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

    def __call__(self, frame, is_last, action: int):
        if (frame is None) or is_last:
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
            frame=self._get_observation(timestep),
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
            frame=timestep.observation,
            is_first=timestep.first(),
            is_last=timestep.last(),
            to_play=0,
            reward=reward,
            legal_actions_mask=legal_actions_mask,
        )


@dataclass
class MinAtarEnvLaw:
    env: dm_env.Environment
    record_video: bool = False

    def __call__(self, is_last, action: int):
        if is_last:
            timestep = self.env.reset()
            if self.record_video:
                pass
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
            frame=self._curr_traj.frame[self._curr_step],
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


@dataclass
class Law:
    name: str
    malloc: Callable[[], Dict[str, Any]]
    apply: Callable[..., Dict[str, Any]]
    read: Set[str]

    @staticmethod
    def from_fn(fn):
        return Law(fn.__name__, lambda: {}, link(fn), get_keys(fn))


def sequential(laws: List[Law]) -> Law:
    name = f"sequential({'+'.join(l.name for l in laws)})"

    def malloc():
        ret = {}
        for l in laws:
            ret.update(l.malloc())
        return ret

    def apply(tape):
        for l in laws:
            with include(tape, l.read) as tape_slice:
                updates = l.apply(tape_slice)
            tape.update(updates)
        return tape

    read: Set[str] = set(sum([list(l.read) for l in laws], []))

    return Law(
        name=name,
        malloc=malloc,
        apply=apply,
        read=read,
    )


def get_keys(fn):
    return inspect.signature(fn).parameters.keys()


# TODO: sort this out
def make_env_law(env_name) -> Law:
    def malloc():
        return {"env": make_env(env_name)}

    def apply(env: dm_env.Environment, is_last: bool, action: int):
        if is_last:
            timestep = env.reset()
        else:
            timestep = env.step(action)

        if timestep.reward is None:
            reward = 0.0
        else:
            reward = timestep.reward

        legal_actions_mask = np.ones(env.action_space.n, dtype=np.float32)

        return dict(
            frame=np.array(timestep.observation, dtype=float),
            is_first=timestep.first(),
            is_last=timestep.last(),
            to_play=0,
            reward=reward,
            legal_actions_mask=legal_actions_mask,
        )

    return Law(
        name="dm_env",
        malloc=malloc,
        # this is not linked yet
        apply=apply,
        read=get_keys(apply),
    )


def make_vec_env(env_name: str, num_envs: int) -> Law:
    def malloc():
        env_laws = [make_env_law(env_name) for _ in range(num_envs)]
        raw_envs: List[dm_env.Environment] = [env.malloc()["env"] for env in env_laws]
        dim_action = raw_envs[0].action_spec().num_values
        frame_shape = raw_envs[0].observation_spec().shape
        # make these allocations stacked from single env setting?
        action = np.full(num_envs, fill_value=0, dtype=jnp.int32)
        action_probs = jnp.full((num_envs, dim_action), fill_value=0, dtype=jnp.float32)
        frame = jnp.zeros((num_envs, *frame_shape), dtype=jnp.float32)
        is_first = jnp.full(num_envs, fill_value=False, dtype=bool)
        is_last = jnp.full(num_envs, fill_value=True, dtype=bool)
        to_play = jnp.zeros(num_envs, dtype=jnp.int32)
        reward = jnp.zeros(num_envs, dtype=jnp.float32)
        legal_actions_mask = jnp.ones((num_envs, dim_action), dtype=jnp.int32)
        return {
            "envs": raw_envs,
            "frame": frame,
            "action": action,
            "action_probs": action_probs,
            "is_first": is_first,
            "is_last": is_last,
            "to_play": to_play,
            "reward": reward,
            "legal_actions_mask": legal_actions_mask,
        }

    env_apply = make_env_law(env_name).apply

    def apply(envs: List[dm_env.Environment], is_last: List[bool], action: List[int]):
        updates_list = []
        for env, is_last_, action_ in zip(envs, is_last, action):
            updates = env_apply(env=env, is_last=is_last_, action=action_)
            updates_list.append(updates)
        return stack_sequence_fields(updates_list)

    return Law(
        name=f"vec_env({env_name} * {num_envs})",
        malloc=malloc,
        apply=link(apply),
        read=get_keys(apply),
    )


def make_stacker(
    num_rows: int,
    num_cols: int,
    num_channels: int,
    num_stacked_frames: int,
    dim_action: int,
):
    def malloc():
        return {
            "stacked_frames": jnp.zeros(
                (num_rows, num_cols, num_stacked_frames * num_channels),
                dtype=jnp.float32,
            ),
            "stacked_actions": jnp.zeros(
                (num_rows, num_cols, num_stacked_frames * dim_action),
                dtype=jnp.float32,
            ),
        }

    def apply(stacked_frames, stacked_actions, frame, action):
        stacked_frames = jnp.append(stacked_frames, frame, axis=-1)
        stacked_frames = stacked_frames[..., np.array(frame.shape[-1]) :]

        action_plane = jax.nn.one_hot(action, dim_action)
        action_plane = jnp.expand_dims(action_plane / dim_action, axis=[0, 1])
        action_plane = jnp.tile(action_plane, (frame.shape[0], frame.shape[1], 1))
        stacked_actions = jnp.append(stacked_actions, action_plane, axis=-1)
        stacked_actions = stacked_actions[..., np.array(action_plane.shape[-1]) :]

        return {"stacked_frames": stacked_frames, "stacked_actions": stacked_actions}

    return Law(
        name=f"batch_frame_stacker({num_rows=}, {num_cols=}, {num_channels=}, {num_stacked_frames=})",
        malloc=malloc,
        apply=link(apply),
        read=get_keys(apply),
    )


def make_batch_stacker(
    batch_size: int,
    num_rows: int,
    num_cols: int,
    num_channels: int,
    num_stacked_frames: int,
    dim_action: int,
):
    stacker = make_stacker(
        num_rows, num_cols, num_channels, num_stacked_frames, dim_action
    )

    def malloc():
        return tree.map_structure(
            lambda x: jnp.repeat(x[None, ...], batch_size, axis=0),
            stacker.malloc(),
        )

    apply = jax.vmap(unlink(stacker.apply))

    return Law(
        name="h",
        malloc=malloc,
        apply=link(apply),
        read=get_keys(apply),
    )


def make_traj_writer(
    num_envs: int,
):
    def malloc():
        # NOTE: mutable state dangerous?
        return {"step_samples": [[] for _ in range(num_envs)]}

    def apply(
        frame,
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
        for i in range(num_envs):
            step_sample = StepSample(
                frame=frame[i],
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

    return Law(
        name=f"traj_writer({num_envs=})",
        malloc=malloc,
        apply=link(apply),
        read=get_keys(apply),
    )


def make_terminator(size: int):
    def malloc():
        return {"quit": False}

    def apply(output_buffer):
        if len(output_buffer) >= size:
            return {"quit": True}
        else:
            return {"quit": False}

    return Law(
        name=f"terminator({size=})",
        malloc=malloc,
        apply=link(apply),
        read=get_keys(apply),
    )


def make_reward_terminator(size: int):
    def malloc():
        return {
            "quit": False,
            "reward_records": tuple(),
            "reward_traj_count": 0,
        }

    def apply(
        reward,
        reward_records,
        reward_traj_count,
        is_last,
    ):
        assert len(is_last) == 1
        reward_records = reward_records + (reward[0],)
        if is_last[0]:
            reward_traj_count += 1

        if reward_traj_count >= size:
            stats = {
                "avr_episode_length": len(reward_records) / reward_traj_count,
                "avr_reward_per_episode": sum(reward_records) / reward_traj_count,
                "avr_reward_per_step": sum(reward_records) / len(reward_records),
            }
            return {
                "quit": True,
                "reward_records": tuple(),
                "reward_traj_count": 0,
                "output_buffer": (stats,),
            }
        else:
            return {
                "quit": False,
                "reward_records": reward_records,
                "reward_traj_count": reward_traj_count,
            }

    return Law(
        name=f"reward_terminator{size=})",
        malloc=malloc,
        apply=link(apply),
        read=get_keys(apply),
    )


def make_min_atar_gif_recorder(n_channels=6, root_dir="gifs"):
    cmap = sns.color_palette("cubehelix", n_channels)
    cmap.insert(0, (0, 0, 0))
    cs = np.array([cmap[i] for i in range(n_channels + 1)])
    font = ImageFont.truetype("courier.ttf", 14)
    root_dir = Path(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    def malloc():
        return {"images": []}

    def apply(
        is_last,
        frame,
        root_value,
        q_values,
        action_probs,
        action,
        reward,
        images: List[Image.Image],
    ):
        numerical_state = np.array(
            np.amax(frame[0] * np.reshape(np.arange(n_channels) + 1, (1, 1, -1)), 2)
            + 0.5,
            dtype=int,
        )
        rgbs = np.array(cs[numerical_state - 1] * 255, dtype=np.uint8)
        img = Image.fromarray(rgbs)
        img = img.resize((img.width * 40, img.height * 40), Image.NEAREST)
        draw = ImageDraw.Draw(img)
        action_map = {
            0: "Stay",
            1: "Left",
            2: "Right",
            3: "Fire",
        }
        with np.printoptions(precision=3, suppress=True, floatmode="fixed"):
            content = (
                f"R {reward[0]}\n"
                f"V {root_value[0]}\n"
                f"π {action_probs[0]}\n"
                f"Q {q_values[0]}\n"
                f"A {action_map[int(action[0])]}\n"
            )
        draw.text((0, 0), content, fill="black", font=font)
        images = images + [img]
        if is_last[0] and images:
            counter = 0
            while gif_fpath := (root_dir / f"{counter}.gif"):
                if gif_fpath.exists():
                    counter += 1
                else:
                    break

            images[0].save(
                str(gif_fpath),
                save_all=True,
                append_images=images[1:],
                optimize=False,
                duration=40,
            )
            logger.info("gif saved to " + str(gif_fpath))
            images = []
        return {"images": images}

    return Law(
        name=f"min_atar_gif_recorder({n_channels=})",
        malloc=malloc,
        apply=link(apply),
        read=get_keys(apply),
    )


def _termination_penalty(is_last, reward):
    reward_overwrite = jax.lax.cond(
        is_last,
        lambda: reward - 1,
        lambda: reward,
    )
    return {"reward": reward_overwrite}

penalty = Law.from_fn(_termination_penalty)
penalty.apply = link(jax.vmap(unlink(penalty.apply)))
