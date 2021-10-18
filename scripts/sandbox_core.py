import collections
import operator
import os
from dataclasses import InitVar, dataclass, field
from functools import partial
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

import acme
import attr
import chex
import dm_env
import jax
import jax.numpy as jnp
import moozi as mz
import numpy as np
import open_spiel
import optax
import ray
import rlax
import tree
import trio
import trio_asyncio
from absl import logging
from acme.utils.tree_utils import unstack_sequence_fields
from acme.wrappers import SinglePrecisionWrapper
from jax._src.numpy.lax_numpy import stack
from moozi.batching_layer import BatchingClient, BatchingLayer
from moozi.learner import TrainingState
from moozi.link import UniverseAsync, link
from moozi.nn import NeuralNetwork
from moozi.replay import Trajectory
from moozi.utils import SimpleBuffer
from trio_asyncio import aio_as_trio

from acme_openspiel_wrapper import OpenSpielWrapper
from config import Config


def make_catch():
    prev_verbosity = logging.get_verbosity()
    logging.set_verbosity(logging.WARNING)

    env_columns, env_rows = 6, 6
    raw_env = open_spiel.python.rl_environment.Environment(
        f"catch(columns={env_columns},rows={env_rows})"
    )
    env = OpenSpielWrapper(raw_env)
    env = SinglePrecisionWrapper(env)
    env_spec = acme.specs.make_environment_spec(env)

    logging.set_verbosity(prev_verbosity)
    return env, env_spec


def make_tic_tac_toe():
    raw_env = open_spiel.python.rl_environment.Environment(f"tic_tac_toe")
    env = OpenSpielWrapper(raw_env)
    env = SinglePrecisionWrapper(env)
    env_spec = acme.specs.make_environment_spec(env)
    return env, env_spec


make_env = make_catch


def get_observation(timestep: dm_env.TimeStep):
    if isinstance(timestep.observation, list):
        assert len(timestep.observation) == 1
        return timestep.observation[0].observation
    else:
        raise NotImplementedError


def get_legal_actions(timestep: dm_env.TimeStep):
    if isinstance(timestep.observation, list):
        assert len(timestep.observation) == 1
        return timestep.observation[0].legal_actions
    else:
        raise NotImplementedError


@dataclass
class Artifact:
    # meta
    universe_id: int = -1
    num_ticks: int = 0
    num_episodes: int = 0
    avg_episodic_reward: float = 0
    sum_episodic_reward: float = 0

    # environment
    timestep: dm_env.TimeStep = None
    obs: np.ndarray = None
    is_first: bool = True
    is_last: bool = False
    to_play: int = -1
    reward: float = 0.0
    action: int = -1
    discount: float = 1.0
    legal_actions_mask: np.ndarray = None

    # planner
    # root: Any = None
    root_value: float = 0
    action_probs: Any = None

    # player
    stacked_frames: np.ndarray = None


@link
@dataclass
class FrameStacker:
    num_frames: int = 1

    padding: Optional[np.ndarray] = None
    deque: collections.deque = None

    def __post_init__(self):
        self.deque = collections.deque(maxlen=self.num_frames)

    def __call__(self, timestep) -> Any:
        if self.padding is None:
            self._make_padding(timestep)

        if timestep.last():
            self.deque.clear()

        self.deque.append(get_observation(timestep))

        stacked_frames = self._get_stacked_frames()

        return dict(stacked_frames=stacked_frames)

    def _get_stacked_frames(self):
        stacked_frames = np.array(list(self.deque))
        num_frames_to_pad = self.num_frames - len(self.deque)
        if num_frames_to_pad > 0:
            paddings = np.stack(
                [np.copy(self.padding) for _ in range(num_frames_to_pad)], axis=0
            )
            stacked_frames = np.append(paddings, np.array(list(self.deque)), axis=0)
        return stacked_frames

    def _make_padding(self, timestep):
        assert timestep.first()
        shape = get_observation(timestep).shape
        self.padding = np.zeros(shape)


@link
def set_legal_actions(timestep):
    return dict(legal_actions_mask=get_legal_actions(timestep))


@link
def wrap_up_episode(timestep, sum_episodic_reward, num_episodes, universe_id):
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
def increment_tick(num_ticks):
    return {"num_ticks": num_ticks + 1}


@link
@dataclass
class EnvironmentLaw:
    env_state: dm_env.Environment

    def __call__(self, timestep: dm_env.TimeStep, action: int):
        if timestep is None or timestep.last():
            timestep = self.env_state.reset()
        else:
            timestep = self.env_state.step([action])
        return dict(
            timestep=timestep,
            obs=get_observation(timestep),
            is_first=timestep.first(),
            is_last=timestep.last(),
            to_play=self.env_state.current_player,
            reward=timestep.reward,
        )


@dataclass(repr=False)
class InteractionManager:
    batching_layers: Optional[List[BatchingLayer]] = None
    universes: Optional[List[UniverseAsync]] = None

    def setup(
        self,
        factory: Callable[[], Tuple[List[BatchingLayer], List[UniverseAsync]]],
    ):
        self.batching_layers, self.universes = factory()
        return self

    def set_verbosity(self, verbosity):
        logging.set_verbosity(verbosity)

    def run(self, num_ticks):

        import trio

        async def main_loop():
            async with trio.open_nursery() as main_nursery:
                for b in self.batching_layers:
                    b.is_paused = False
                    main_nursery.start_soon(b.start_processing)
                    main_nursery.start_soon(b.start_logging)

                async with trio.open_nursery() as universe_nursery:
                    for u in self.universes:
                        universe_nursery.start_soon(partial(u.tick, times=num_ticks))
                for b in self.batching_layers:
                    b.is_paused = True

        trio_asyncio.run(main_loop)
        return [u.artifact for u in self.universes]


@link
def set_random_action_from_timestep(timestep: dm_env.TimeStep):
    if not timestep.last():
        legal_actions = timestep.observation[0].legal_actions
        random_action = np.random.choice(np.flatnonzero(legal_actions == 1))
        return {"action": random_action}
    else:
        return {"action": -1}


# @link
# def set_action_probs():
#     action_probs = np.zeros_like(mcts.all_actions_mask, dtype=np.float32)
#     for a, visit_count in mcts_tree.get_children_visit_counts().items():
#         action_probs[a] = visit_count
#     action_probs /= np.sum(action_probs)


def make_sgd_step_fn(network: mz.nn.NeuralNetwork, loss_fn: mz.loss.LossFn, optimizer):
    @partial(jax.jit, backend="cpu")
    @chex.assert_max_traces(n=1)
    def sgd_step_fn(training_state: TrainingState, batch: mz.replay.TrainTarget):
        # gradient descend
        _, new_key = jax.random.split(training_state.rng_key)
        grads, extra = jax.grad(loss_fn, has_aux=True, argnums=1)(
            network, training_state.params, batch
        )
        updates, new_opt_state = optimizer.update(grads, training_state.opt_state)
        new_params = optax.apply_updates(training_state.params, updates)
        steps = training_state.steps + 1
        new_training_state = TrainingState(new_params, new_opt_state, steps, new_key)

        # KL calculation
        orig_logits = network.initial_inference(
            training_state.params, batch.stacked_frames
        ).policy_logits
        new_logits = network.initial_inference(
            new_params, batch.stacked_frames
        ).policy_logits
        prior_kl = jnp.mean(rlax.categorical_kl_divergence(orig_logits, new_logits))

        # store data to log
        step_data = mz.logging.JAXBoardStepData(scalars={}, histograms={})
        step_data.update(extra)
        step_data.scalars["prior_kl"] = prior_kl
        # step_data.histograms["reward"] = batch.last_reward
        step_data.add_hk_params(new_params)

        return new_training_state, step_data

    return sgd_step_fn


@dataclass(repr=False)
class InferenceServer:
    network: InitVar
    params: InitVar
    loss_fn: InitVar[mz.loss.LossFn]
    optimizer: InitVar[optax.GradientTransformation]

    state: TrainingState = field(init=False)

    init_inf_fn: Callable = field(init=False)
    recurr_inf_fn: Callable = field(init=False)

    sgd_step_fn: Callable = field(init=False)

    def __post_init__(self, network, params, loss_fn, optimizer):
        logging.info("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
        logging.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
        logging.info(f"jax.devices(): {jax.devices()}")

        # BUG: use the lastest params
        self.init_inf_fn = jax.jit(network.initial_inference)
        self.recurr_inf_fn = jax.jit(network.recurrent_inference)
        self.state = TrainingState(
            params=params,
            opt_state=optimizer.init(params),
            steps=0,
            rng_key=jax.random.PRNGKey(0),
        )
        self.sgd_step_fn = make_sgd_step_fn(network, loss_fn, optimizer)

    def init_inf(self, frames):
        batch_size = len(frames)
        # TODO: concatnating frames should be on the client side?
        results = self.init_inf_fn(self.state.params, np.array(frames))
        results = tree.map_structure(np.array, results)
        return unstack_sequence_fields(results, batch_size)

    def recurr_inf(self, inputs):
        batch_size = len(inputs)
        hidden_states = np.array(list(map(operator.itemgetter(0), inputs)))
        actions = np.array(list(map(operator.itemgetter(1), inputs)))
        results = self.recurr_inf_fn(self.state.params, hidden_states, actions)
        results = tree.map_structure(np.array, results)
        return unstack_sequence_fields(results, batch_size)

    def update(self, batch):
        self.state, step_data = self.sgd_step_fn(self.state, batch)
        return step_data

    @staticmethod
    def make_init_inf_remote_fn(handler):
        """For using with Ray."""

        async def init_inf_remote(x):
            return await aio_as_trio(handler.init_inf.remote(x))

        return init_inf_remote

    @staticmethod
    def make_recurr_inf_remote_fn(handler):
        """For using with Ray."""

        async def recurr_inf_remote(x):
            return await aio_as_trio(handler.recurr_inf.remote(x))

        return recurr_inf_remote


def setup_config(config: Config):
    def make_artifact(index):
        return Artifact(universe_id=index)

    config.set(Config.ARTIFACT_FACTORY, make_artifact)

    config.set(Config.ENV_FACTORY, lambda: make_catch()[0])
    config.set(Config.ENV_SPEC, make_catch()[1])
    config.set(Config.NUM_UNIVERSES, 2000)
