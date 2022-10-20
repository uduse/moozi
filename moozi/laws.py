import collections
import chex
import inspect
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

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
from PIL import Image, ImageDraw, ImageFont, ImageOps

from moozi.core import BASE_PLAYER, PolicyFeed, StepSample
from moozi.core.utils import (
    make_frame_planes,
    make_one_hot_planes,
    push_and_rotate_out_planes,
    fifo_append,
)
from moozi.core.env import _make_dm_env
from moozi.core.link import link, unlink
from moozi.core.tape import include
from moozi.core.types import TrainTarget, TrajectorySample


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


@dataclass
class Law:
    name: str
    malloc: Callable[[], Dict[str, Any]]
    apply: Callable[..., Dict[str, Any]]
    read: Set[str]

    @staticmethod
    def wrap(fn):
        return Law(fn.__name__, malloc=lambda: {}, apply=link(fn), read=get_keys(fn))

    def jit(self, max_trace: int = 1, **kwargs) -> "Law":
        apply = chex.assert_max_traces(n=max_trace)(self.apply)
        apply = jax.jit(apply, **kwargs)
        return Law(self.name, self.malloc, apply, self.read)

    def vmap(self, batch_size, **kwargs) -> "Law":
        name = self.name + f"[vmap * {batch_size}]"
        malloc = jax.vmap(self.malloc, axis_size=batch_size)
        apply = link(jax.vmap(unlink(self.apply), **kwargs, axis_size=batch_size))
        return Law(name, malloc, apply, self.read)


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
        return {"env": _make_dm_env(env_name)}

    def apply(env: dm_env.Environment, is_last: bool, action: int):
        if is_last:
            timestep = env.reset()
        else:
            # action 0 is reserved for termination
            timestep = env.step(action - 1)

        if timestep.reward is None:
            reward = 0.0
        else:
            reward = timestep.reward

        legal_actions_mask = np.ones(env.action_space.n, dtype=np.float32)

        return dict(
            frame=np.asarray(timestep.observation, dtype=float),
            is_first=np.asarray(timestep.first(), dtype=bool),
            is_last=np.asarray(timestep.last(), dtype=bool),
            to_play=np.asarray(0, dtype=int),
            reward=np.asarray(reward, dtype=float),
            legal_actions_mask=np.asarray(legal_actions_mask, dtype=int),
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
        stacked_actions = jnp.zeros(
            (num_rows, num_cols, num_stacked_frames * dim_action),
            dtype=jnp.float32,
        )
        stacked_actions = stacked_actions.at[:, :, 0].set(1)
        stacked_actions = stacked_actions / dim_action

        return {
            "stacked_frames": jnp.zeros(
                (num_rows, num_cols, num_stacked_frames * num_channels),
                dtype=jnp.float32,
            ),
            "stacked_actions": stacked_actions,
        }

    def apply(stacked_frames, stacked_actions, frame, action, is_first):
        def _stack(stacked_frames, stacked_actions, frame, action):
            new_stacked_frames = jnp.append(stacked_frames, frame, axis=-1)
            frame_plane_size = np.array(frame.shape[-1])
            new_stacked_frames = new_stacked_frames[..., frame_plane_size:]
            action_plane = jax.nn.one_hot(action, dim_action)
            action_plane = jnp.expand_dims(action_plane / dim_action, axis=[0, 1])
            action_plane = jnp.tile(action_plane, (frame.shape[0], frame.shape[1], 1))
            new_stacked_actions = jnp.append(stacked_actions, action_plane, axis=-1)
            action_plane_size = np.array(action_plane.shape[-1])
            new_stacked_actions = new_stacked_actions[..., action_plane_size:]
            return {
                "stacked_frames": new_stacked_frames,
                "stacked_actions": new_stacked_actions,
            }

        def _reset_and_stack(stacked_frames, stacked_actions, frame, action):
            ret = malloc()
            return _stack(**ret, frame=frame, action=action)

        return jax.lax.cond(
            is_first,
            _reset_and_stack,
            _stack,
            stacked_frames,
            stacked_actions,
            frame,
            action,
        )

    return Law(
        name=f"batch_frame_stacker({num_rows=}, {num_cols=}, {num_channels=}, {num_stacked_frames=})",
        malloc=malloc,
        apply=link(apply),
        read=get_keys(apply),
    )


# @Law.wrap
# def reset_stacked_frames_and_actions_if_last(is_last, stacked_frames, stacked_actions):
#     return jax.lax.cond(
#         is_last,
#         lambda: {
#             "stacked_frames": jnp.zeros_like(stacked_frames),
#             "stacked_actions": jnp.zeros_like(stacked_actions),
#         },
#         lambda: {
#             "stacked_frames": stacked_frames,
#             "stacked_actions": stacked_actions,
#         },
#     )


def make_batch_stacker_v2(
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
    return stacker.vmap(batch_size)


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
        name="batch_stacker",
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


def make_output_buffer_waiter(size: int):
    def apply(output_buffer):
        if len(output_buffer) >= size:
            return {
                "output": output_buffer[:size],
                "output_buffer": output_buffer[size:],
            }

    return Law(
        name=f"buffer_waiter({size=})",
        malloc=lambda: {},
        apply=link(apply),
        read=get_keys(apply),
    )


def make_steps_waiter(steps: int):
    def malloc():
        return {"curr_steps": 0}

    def apply(curr_steps, output_buffer):
        if curr_steps >= steps:
            return {
                "curr_steps": 0,
                "output": output_buffer,
                "output_buffer": tuple(),
            }
        else:
            return {"curr_steps": curr_steps + 1}

    return Law(
        name="steps_waiter",
        malloc=malloc,
        apply=link(apply),
        read=get_keys(apply),
    )


def make_reward_terminator(size: int):
    def malloc():
        return {
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
                "eval/episode_length": len(reward_records) / reward_traj_count,
                "eval/episode_return": sum(reward_records) / reward_traj_count,
                "eval/reward_per_step": sum(reward_records) / len(reward_records),
            }
            return {
                "output": stats,
                "reward_records": tuple(),
                "reward_traj_count": 0,
            }
        else:
            return {
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
    vis = MinAtarVisualizer(num_channels=n_channels)

    def malloc():
        return {"images": []}

    def apply(
        is_last,
        frame,
        root_value,
        q_values,
        action_probs,
        prior_probs,
        action,
        reward,
        images: List[Image.Image],
        visit_counts,
    ):
        image = vis.make_image(
            frame=frame[0],
        )
        image = vis.add_descriptions(
            image,
            root_value=root_value[0],
            q_values=q_values[0],
            action_probs=action_probs[0],
            prior_probs=prior_probs[0],
            action=action[0],
            reward=reward[0],
            visit_counts=visit_counts[0],
        )
        images = images + [image]
        if is_last[0]:
            vis.save_gif(images, root_dir)
            images = []
        return {"images": images}

    return Law(
        name=f"min_atar_gif_recorder({n_channels=})",
        malloc=malloc,
        apply=link(apply),
        read=get_keys(apply),
    )


@Law.wrap
def concat_stacked_to_obs(stacked_frames, stacked_actions):
    return {"obs": jnp.concatenate([stacked_frames, stacked_actions], axis=-1)}


def make_env_mocker():
    def malloc():
        return {"traj": None, "curr_traj_index": 0}

    def apply(traj: StepSample, curr_traj_index: int):
        chex.assert_rank(traj.frame, 4)
        traj_len = traj.frame.shape[0]
        last_step = curr_traj_index == (traj_len - 1)
        ret = {
            "curr_traj_index": curr_traj_index + 1,
            "frame": np.expand_dims(traj.frame[curr_traj_index], axis=0),
            "action": np.expand_dims(traj.action[curr_traj_index], axis=0),
            "reward": np.expand_dims(traj.last_reward[curr_traj_index], axis=0),
            "legal_actions_mask": np.expand_dims(
                traj.legal_actions_mask[curr_traj_index], axis=0
            ),
            "to_play": np.expand_dims(traj.to_play[curr_traj_index], axis=0),
            "is_first": np.expand_dims(traj.is_first[curr_traj_index], axis=0),
            "is_last": np.expand_dims(traj.is_last[curr_traj_index], axis=0),
        }
        if last_step:
            ret["traj"] = None
            ret["curr_traj_index"] = 0
        return ret

    return Law(
        name="env_mocker",
        malloc=malloc,
        apply=link(apply),
        read=get_keys(apply),
    )


def make_last_step_penalizer(penalty: float = 10.0):
    @Law.wrap
    def penalize_last_step(is_last, reward):
        reward_overwrite = jax.lax.cond(
            is_last,
            lambda: reward - penalty,
            lambda: reward,
        )
        return {"reward": reward_overwrite}

    return penalize_last_step


class MinAtarVisualizer:
    def __init__(self, num_channels: int = 6):
        self._num_channels = num_channels
        cmap = sns.color_palette("cubehelix", self._num_channels)
        cmap.insert(0, (0, 0, 0))
        self.colors = np.array([cmap[i] for i in range(self._num_channels + 1)])
        try:
            self.font = ImageFont.truetype("courier.ttf", 10)
        except:
            font_path_user_root = str(Path("~/courier.ttf").expanduser().resolve())
            self.font = ImageFont.truetype(font_path_user_root, 10)

    def make_image(
        self,
        frame,
    ) -> Image:
        frame = np.asarray(frame, dtype=np.float32)
        numerical_state = np.array(
            np.amax(
                frame * np.reshape(np.arange(self._num_channels) + 1, (1, 1, -1)), 2
            )
            + 0.5,
            dtype=int,
        )
        rgbs = np.array(self.colors[numerical_state - 1] * 255, dtype=np.uint8)
        img = Image.fromarray(rgbs)
        img = img.resize((img.width * 25, img.height * 25), Image.Resampling.NEAREST)
        return img

    def add_descriptions(
        self,
        img,
        root_value=None,
        q_values=None,
        action_probs=None,
        prior_probs=None,
        action=None,
        reward=None,
        action_map=None,
        visit_counts=None,
        n_step_return=None,
    ):
        if root_value is not None:
            root_value = np.asarray(root_value, dtype=np.float32)
        if q_values is not None:
            q_values = np.asarray(q_values, dtype=np.float32)
        if action_probs is not None:
            action_probs = np.asarray(action_probs, dtype=np.float32)
        if action is not None:
            action = np.asarray(action, dtype=np.int32)
        if reward is not None:
            reward = np.asarray(reward, dtype=np.float32)
        if visit_counts is not None:
            visit_counts = np.asarray(visit_counts, dtype=np.int32)
        if prior_probs is not None:
            prior_probs = np.asarray(prior_probs, dtype=np.float32)
        if n_step_return is not None:
            n_step_return = np.asarray(n_step_return, dtype=np.float32)

        img = img.copy()
        draw = ImageDraw.Draw(img)
        with np.printoptions(precision=3, suppress=True, floatmode="fixed"):
            content = ""
            if reward is not None:
                content += f"R {reward}\n"
            if n_step_return is not None:
                content += f"G {n_step_return}\n"
            if root_value is not None:
                content += f"V {root_value:.3f}\n"
            if prior_probs is not None:
                content += f"P {prior_probs}\n"
            if visit_counts is not None:
                content += f"N {visit_counts}\n"
            if action_probs is not None:
                content += f"π {action_probs}\n"
            if q_values is not None:
                content += f"Q {q_values}\n"
            if action is not None:
                if action_map:
                    action = action_map[action]
                content += f"A {action}\n"
        draw.text((0, 0), content, fill="black", font=self.font)
        return img

    def cat_images(self, images, max_columns: int = 5, border: bool = True):
        if border:
            images = [
                ImageOps.expand(image, border=(2, 2, 2, 2), fill="black")
                for image in images
            ]

        width, height = images[0].width, images[0].height
        num_columns = min(len(images), max_columns)
        num_rows = int(np.ceil(len(images) / max_columns))
        canvas_width = width * num_columns
        canvas_height = height * num_rows
        dst = Image.new("RGB", (canvas_width, canvas_height))
        for row in range(num_rows):
            for col in range(num_columns):
                idx = row * num_columns + col
                if idx >= len(images):
                    continue
                dst.paste(images[idx], (col * width, row * height))
        return dst

    def save_gif(self, images, root_dir):
        root_dir = Path(root_dir)
        root_dir.mkdir(parents=True, exist_ok=True)
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
            optimize=True,
            quality=40,
            duration=40,
        )
        logger.info("gif saved to " + str(gif_fpath))

    def make_images_from_obs(self, obs, num_stacked_frames) -> list:
        images = []
        for i in range(num_stacked_frames):
            lower = self._num_channels * i
            upper = self._num_channels * (i + 1)
            img = self.make_image(obs[:, :, lower:upper])
            img = self.add_descriptions(img)
            images.append(img)
        return images


def make_obs_processor(
    num_rows: int,
    num_cols: int,
    num_channels: int,
    num_stacked_frames: int,
    dim_action: int,
):
    def malloc():
        empty_frames = jnp.zeros(
            (num_stacked_frames, num_rows, num_cols, num_channels),
            dtype=jnp.float32,
        )
        empty_actions = jnp.zeros(
            (num_stacked_frames,),
            dtype=jnp.int32,
        )
        stacked_frame_planes = make_frame_planes(empty_frames)
        stacked_action_planes = make_one_hot_planes(
            empty_actions, num_rows, num_cols, dim_action
        )
        empty_obs = jnp.concatenate(
            [stacked_frame_planes, stacked_action_planes],
            axis=-1,
        )
        return {
            "history_frames": empty_frames,
            "history_actions": empty_actions,
            "obs": empty_obs,
        }

    def apply(history_frames, history_actions, frame, action, is_first):
        def _make_obs(history_frames, history_actions, frame, action):
            history_frames = fifo_append(history_frames, frame)
            history_actions = fifo_append(history_actions, action)
            stacked_frame_planes = make_frame_planes(history_frames)
            stacked_action_planes = make_one_hot_planes(
                history_actions, num_rows, num_cols, dim_action
            )
            obs = jnp.concatenate(
                [stacked_frame_planes, stacked_action_planes],
                axis=-1,
            )
            return {
                "history_frames": history_frames,
                "history_actions": history_actions,
                "obs": obs,
            }

        def _reset_then_make_obs(history_frames, history_actions, frame, action):
            ret = malloc()
            return _make_obs(
                history_frames=ret["history_frames"],
                history_actions=ret["history_actions"],
                frame=frame,
                action=action,
            )

        return jax.lax.cond(
            is_first,
            _reset_then_make_obs,
            _make_obs,
            history_frames,
            history_actions,
            frame,
            action,
        )

    return Law(
        name="obs_processor",
        malloc=malloc,
        apply=link(apply),
        read=get_keys(apply),
    )


def make_batch_env_mocker(batch_size):
    def malloc():
        return {"dispatcher": TrajectoryDispatcher(batch_size), "trajs": tuple()}

    def apply(
        dispatcher: TrajectoryDispatcher,
        trajs: Tuple[TrajectorySample, ...],
    ):
        dispatcher.add_trajs(list(trajs))
        try:
            dispatcher.refill()
        except ValueError:
            logger.warning(
                "Reanalyze early termination due to insufficient trajectory supply."
            )
            return {"dispatcher": dispatcher, "trajs": tuple(), "curr_step": 1000000}
        assert dispatcher.get_safe_buffer_refill_count() <= batch_size
        ret = dispatcher.step()
        return {"dispatcher": dispatcher, "trajs": tuple(), **ret}

    return Law(
        name="batch_env_mocker",
        malloc=malloc,
        apply=link(apply),
        read=get_keys(apply),
    )


class TrajectoryDispatcher:
    def __init__(self, batch_size) -> None:
        self._buffer: Deque[TrajectorySample] = collections.deque(maxlen=batch_size * 2)
        self._mockers: Dict[int, Optional["TrajectoryMocker"]] = {
            i: None for i in range(batch_size)
        }

    def add_trajs(self, trajs: Sequence[TrajectorySample]) -> None:
        if trajs:
            self._buffer.extend(trajs)
            logger.debug(f"{len(trajs)} trajectories moved to buffer")

    def step(self) -> dict:
        steps = []
        for i, mocker in self._mockers.items():
            assert mocker is not None
            steps.append(mocker.step())
            if mocker.is_done:
                self._mockers[i] = None

        return stack_sequence_fields(steps)

    def refill(self) -> None:
        to_refill = self._get_to_refill_ids()

        if len(to_refill) > len(self._buffer):
            raise ValueError

        for i in to_refill:
            self._mockers[i] = TrajectoryMocker(self._buffer.pop())

        logger.debug(f"{len(to_refill)} trajectories refilled")

    def get_safe_buffer_refill_count(self) -> int:
        safe_buffer_size = len(self._mockers)
        curr_buffer_size = len(self._buffer)
        num_empty = len(self._get_to_refill_ids())
        return max(safe_buffer_size - curr_buffer_size + num_empty, 0)

    def _get_to_refill_ids(self) -> List[int]:
        to_refill = []
        for i, m in self._mockers.items():
            if m is None:
                to_refill.append(i)
        return to_refill


def make_step_counter():
    def malloc():
        return {"step_count": 0}

    def apply(step_count):
        return {"step_count": step_count + 1}

    return Law(
        name="step_counter",
        malloc=malloc,
        apply=link(apply),
        read=get_keys(apply),
    )


@dataclass
class TrajectoryMocker:
    traj: TrajectorySample
    idx: int = 0

    def __post_init__(self):
        chex.assert_rank(self.traj.frame, 4)

    def step(self) -> dict:
        if self.idx == -1:
            raise ValueError

        curr_slice = self.get_curr_slice()

        ret = {
            "frame": curr_slice.frame,
            "action": curr_slice.action,
            "reward": curr_slice.last_reward,
            "legal_actions_mask": curr_slice.legal_actions_mask,
            "to_play": curr_slice.to_play,
            "is_first": curr_slice.is_first,
            "is_last": curr_slice.is_last,
        }

        if curr_slice.is_last:
            is_actually_last_frame = self.idx == (self.traj.frame.shape[0] - 1)
            assert is_actually_last_frame
            self.idx = -1
        else:
            self.idx += 1

        return ret

    def get_curr_slice(self) -> StepSample:
        return tree.map_structure(lambda x: x.take(self.idx, axis=0), self.traj)

    @property
    def is_done(self):
        return self.idx == -1
        
