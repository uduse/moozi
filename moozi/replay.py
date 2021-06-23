import copy
import random
from typing import Iterable, List, NamedTuple, Optional

import chex
import dm_env
import numpy as np
import reverb
import tensorflow as tf
import tree
from acme import datasets, specs, types
from acme.adders.reverb import DEFAULT_PRIORITY_TABLE, ReverbAdder
from acme.agents.replay import ReverbReplay
from acme.utils import tree_utils
from acme.wrappers import open_spiel_wrapper
from nptyping import NDArray

import moozi as mz


class Observation(NamedTuple):
    frame: np.ndarray
    reward: np.float32
    # legal_actions_mask: chex.Array
    is_first: np.bool
    is_last: np.bool

    @staticmethod
    def from_env_timestep(timestep: dm_env.TimeStep):
        if isinstance(timestep.observation, open_spiel_wrapper.OLT):
            frame = timestep.observation.observation.astype(np.float32)
            if timestep.reward is None:
                reward = np.float32(0)
            else:
                reward = np.float32(timestep.reward).squeeze()
            # legal_actions_mask = timestep.observation[0].legal_actions.astype(np.bool)
            is_first = np.bool(timestep.first())
            is_last = np.bool(timestep.last())
            # return Observation(frame, reward, legal_actions_mask, is_first, is_last)
            return Observation(frame, reward, is_first, is_last)
        else:
            raise NotImplementedError


class Reflection(NamedTuple):
    action: np.ndarray
    root_value: np.ndarray
    child_visits: np.ndarray


def _prefix_dim(spec, size):
    return spec.replace(shape=(size,) + spec.shape)


def make_signature(env_spec: specs.EnvironmentSpec, max_episode_length):

    obs_signature = Observation(
        frame=_prefix_dim(env_spec.observations.observation, max_episode_length),
        reward=_prefix_dim(env_spec.rewards, max_episode_length),
        is_first=specs.Array(shape=(max_episode_length,), dtype=np.bool),
        is_last=specs.Array(shape=(max_episode_length,), dtype=np.bool),
    )

    ref_signature = Reflection(
        action=specs.Array(shape=(max_episode_length,), dtype=np.int32),
        root_value=specs.Array(shape=(max_episode_length,), dtype=np.float32),
        child_visits=specs.Array(
            shape=(max_episode_length, env_spec.actions.num_values), dtype=np.float32
        ),
    )

    merged_signature = {**obs_signature._asdict(), **ref_signature._asdict()}
    # return tree.map_structure(tf.TensorSpec.from_spec, merged_signature)
    return tree.map_structure_with_path(
        lambda p, v: tf.TensorSpec.from_spec(v, p[0]), merged_signature
    )


class Adder(ReverbAdder):
    def __init__(
        self,
        client: reverb.Client,
        # num_unroll_steps: int = 5,
        # num_stacked_frames: int = 1,
        # num_td_steps: int = 1000,
        # discount: float = 1,
        delta_encoded: bool = False,
        # according to the pseudocode, 500 is roughly enough for board games
        max_episode_length: int = 500,
        max_inflight_items: int = 1,
    ):

        self._client = client
        # self._num_unroll_steps = num_unroll_steps
        # self._num_td_steps = num_td_steps
        # self._num_stacked_frames = num_stacked_frames
        # self._max_episode_length = max_episode_length

        super().__init__(
            client=client,
            max_sequence_length=max_episode_length,
            max_in_flight_items=max_inflight_items,
            delta_encoded=delta_encoded,
        )

    def add_first(self, observation: Observation):
        assert observation.is_first
        assert not observation.is_last
        assert np.isclose(observation.reward, 0)

        self._writer.append(observation._asdict(), partial_step=True)

    def add(self, last_reflection: Reflection, next_observation: Observation):
        try:
            _ = self._writer.history
        except RuntimeError:
            raise ValueError("adder.add_first must be called before adder.add.")

        self._writer.append(last_reflection._asdict(), partial_step=False)
        self._writer.append(next_observation._asdict(), partial_step=True)

        if next_observation.is_last:
            # pad the incomplete last step
            partial_padding_step = tree.map_structure(
                np.zeros_like, last_reflection
            )._asdict()
            self._writer.append(partial_padding_step, partial_step=False)

            self._write_last()
            self.reset()

    def _write(self):
        # This adder only writes at the end of the episode, see _write_last()
        pass

    def _write_last(self):
        r"""
        There are two ways of storing experiences for training.
        One way is to store the entire episode into the replay and create
        """
        trajectory = tree.map_structure(lambda x: x[:], self._writer.history)
        self._writer.create_item(DEFAULT_PRIORITY_TABLE, 1.0, trajectory)


class Trajectory(NamedTuple):
    frame: NDArray[np.float32]
    reward: NDArray[np.float32]
    is_first: NDArray[np.bool]
    is_last: NDArray[np.bool]
    action: NDArray[np.int32]
    root_value: NDArray[np.float32]
    child_visits: NDArray[np.float32]

    def cast(self) -> "Trajectory":
        return Trajectory(
            frame=np.asarray(self.frame, dtype=np.float32),
            reward=np.asarray(self.reward, dtype=np.float32),
            is_first=np.asarray(self.is_first, dtype=np.bool),
            is_last=np.asarray(self.is_last, dtype=np.bool),
            action=np.asarray(self.action, dtype=np.int32),
            root_value=np.asarray(self.root_value, dtype=np.float32),
            child_visits=np.asarray(self.child_visits, dtype=np.float32),
        )


class TrainTarget(NamedTuple):
    stacked_frames: NDArray[np.float32]
    action: NDArray[np.int32]
    value: NDArray[np.float32]
    last_reward: NDArray[np.float32]
    child_visits: NDArray[np.float32]

    def cast(self) -> "TrainTarget":
        return TrainTarget(
            stacked_frames=np.asarray(self.stacked_frames, dtype=np.float32),
            action=np.asarray(self.action, dtype=np.int32),
            value=np.asarray(self.value, dtype=np.float32),
            last_reward=np.asarray(self.last_reward, dtype=np.float32),
            child_visits=np.asarray(self.child_visits, dtype=np.float32),
        )


def make_target(
    sample: Trajectory,
    start_idx,
    discount,
    num_unroll_steps,
    num_td_steps,
    num_stacked_frames,
):
    # assert not batched
    assert len(sample.reward.shape) == 1

    last_step_idx = sample.is_last.argmax()

    # unroll
    unrolled_data = []
    for curr_idx in range(start_idx, start_idx + num_unroll_steps + 1):
        value = _get_value(sample, curr_idx, last_step_idx, num_td_steps, discount)
        last_reward = _get_last_reward(sample, start_idx, curr_idx, last_step_idx)
        child_visits = _get_child_visits(sample, curr_idx, last_step_idx)
        unrolled_data.append((value, last_reward, child_visits))

    unrolled_data_stacked = tree_utils.stack_sequence_fields(unrolled_data)

    stacked_frames = _get_stacked_frames(sample, start_idx, num_stacked_frames)
    action = _get_action(sample, start_idx, num_unroll_steps)

    return TrainTarget(
        stacked_frames=stacked_frames,
        action=action,
        value=unrolled_data_stacked[0],
        last_reward=unrolled_data_stacked[1],
        child_visits=unrolled_data_stacked[2],
    )


def _get_action(sample, start_idx, num_unroll_steps):
    action = sample.action[start_idx : start_idx + num_unroll_steps]
    num_actions_to_pad = num_unroll_steps - action.size
    if num_actions_to_pad > 0:
        action = np.concatenate((action, np.full(num_actions_to_pad, -1)))
    return action


def _get_stacked_frames(sample, start_idx, num_stacked_frames):
    frame_idx_lower = max(start_idx - num_stacked_frames + 1, 0)
    frame = sample.frame[frame_idx_lower : start_idx + 1]
    num_frames_to_pad = num_stacked_frames - frame.shape[0]
    if num_frames_to_pad > 0:
        padding_shape = (num_frames_to_pad,) + frame.shape[1:]
        frame = np.concatenate((np.zeros(shape=padding_shape), frame))
    return frame


def _get_value(sample, curr_idx, last_step_idx, num_td_steps, discount):
    bootstrap_idx = curr_idx + num_td_steps
    if bootstrap_idx <= last_step_idx:
        value = sample.root_value[bootstrap_idx] * discount ** num_td_steps
    else:
        value = 0

    for i, reward in enumerate(sample.reward[curr_idx + 1 : bootstrap_idx + 1]):
        value += reward * discount ** i
    return value


def _get_last_reward(sample, start_idx, curr_idx, last_step_idx):
    if curr_idx == start_idx:
        return 0
    elif curr_idx <= last_step_idx:
        return sample.reward[curr_idx]
    else:
        return 0


def _get_child_visits(sample, curr_idx, last_step_idx):
    if curr_idx <= last_step_idx:
        return sample.child_visits[curr_idx]
    else:
        return np.zeros_like(sample.child_visits[0])


def make_replay(
    env_spec: specs.EnvironmentSpec,
    max_episode_length: int = 500,
    batch_size: int = 32,
    max_replay_size: int = 100_000,
    min_replay_size: int = 1,
    prefetch_size: int = 4,
    replay_table_name: str = DEFAULT_PRIORITY_TABLE,
) -> ReverbReplay:

    signature = mz.replay.make_signature(env_spec, max_episode_length)

    replay_table = reverb.Table(
        name=replay_table_name,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        max_size=max_replay_size,
        rate_limiter=reverb.rate_limiters.MinSize(min_replay_size),
        signature=signature,
    )
    server = reverb.Server([replay_table], port=None)

    # The adder is used to insert observations into replay.
    address = f"localhost:{server.port}"
    client = reverb.Client(address)
    adder = Adder(client, max_episode_length=max_episode_length, delta_encoded=True)

    # The dataset provides an interface to sample from replay.
    data_iterator = datasets.make_reverb_dataset(
        table=replay_table_name,
        server_address=address,
        batch_size=batch_size,
        prefetch_size=prefetch_size,
    ).as_numpy_iterator()

    return ReverbReplay(
        server=server, adder=adder, data_iterator=data_iterator, client=client
    )


def post_process_data_iterator(
    data_iterator,
    batch_size: int,
    discount: float,
    num_unroll_steps: int,
    num_td_steps: int,
    num_stacked_frames: int,
):
    # NOTE: use jax.random instead of python native random?

    def _make_one_target_from_one_trajectory(traj: Trajectory):
        last_step_idx = traj.is_last.argmax()
        random_start = random.randrange(last_step_idx + 1)
        return mz.replay.make_target(
            traj,
            random_start,
            discount,
            num_unroll_steps,
            num_td_steps,
            num_stacked_frames,
        )

    while data := next(data_iterator).data:
        raw_trajs = tree_utils.unstack_sequence_fields(data, batch_size=batch_size)
        trajs = list(map(lambda x: Trajectory(**x), raw_trajs))
        batched_target = tree_utils.stack_sequence_fields(
            map(_make_one_target_from_one_trajectory, trajs)
        )
        yield batched_target
