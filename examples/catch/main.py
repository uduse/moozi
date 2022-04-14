# %%
import os
from unicodedata import name

import haiku as hk
import jax
import optax
import tree
from moozi.loss import MuZeroLoss
from moozi.nn.nn import RootFeatures, make_model
from moozi.nn.resnet import ResNetArchitecture, ResNetSpec

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from functools import partial
from typing import List, NamedTuple

import moozi as mz
import numpy as np
import ray
from absl import logging
from moozi.core import Config, UniverseAsync, link
from moozi.core.env import make_env
from moozi.laws import (
    OpenSpielEnvLaw,
    FrameStacker,
    ReanalyzeEnvLaw,
    TrajectoryOutputWriter,
    exit_if_no_input,
    make_policy_feed,
    output_last_step_reward,
    update_episode_stats,
)
from moozi.logging import (
    JAXBoardLoggerActor,
    JAXBoardLoggerV2,
    LogScalar,
    TerminalLogger,
    TerminalLoggerActor,
)
from moozi.parameter_optimizer import ParameterOptimizer
from moozi.mcts import ActionSamplerLaw, Planner
from moozi.replay import ReplayBuffer
from moozi.rollout_worker import make_rollout_workers
from moozi.utils import WallTimer


# %%
ray.init(address='auto', _redis_password='5241590000000000')

# %%
game_num_rows = 6
game_num_cols = 6
num_epochs = 10

config = Config()
config.env = f"catch(rows={game_num_rows},columns={game_num_cols})"

config.big_batch_size = 256
config.batch_size = 32

config.discount = 0.99
config.num_unroll_steps = 2
config.num_td_steps = 100
config.num_stacked_frames = 1
config.lr = 3e-3

config.replay_max_size = 100000
config.replay_min_size = 1
config.replay_prefetch_max_size = 2048

config.num_epochs = num_epochs
config.num_ticks_per_epoch = game_num_rows
config.num_updates_per_samples_added = 30

config.num_env_workers = 2
config.num_universes_per_env_worker = 20

config.num_reanalyze_workers = 4
config.num_universes_per_reanalyze_worker = 20
config.num_trajs_per_reanalyze_universe = 1

config.weight_decay = 5e-2
config.known_bound_min = -1
config.known_bound_max = 1
config.nn_arch_cls = mz.nn.ResNetArchitecture

env_spec = mz.make_env_spec(config.env)
single_frame_shape = env_spec.observations.observation.shape
obs_channels = single_frame_shape[-1] * config.num_stacked_frames
repr_channels = 4
dim_action = env_spec.actions.num_values

config.nn_spec = mz.nn.ResNetSpec(
    obs_rows=game_num_rows,
    obs_cols=game_num_cols,
    obs_channels=obs_channels,
    repr_rows=game_num_rows,
    repr_cols=game_num_cols,
    repr_channels=repr_channels,
    dim_action=dim_action,
    repr_tower_blocks=4,
    repr_tower_dim=4,
    pred_tower_blocks=4,
    pred_tower_dim=4,
    dyna_tower_blocks=4,
    dyna_tower_dim=4,
    dyna_state_blocks=4,
)
config.print()

num_interactions = (
    config.num_epochs
    * config.num_ticks_per_epoch
    * config.num_env_workers
    * config.num_universes_per_env_worker
)
print(f"num_interactions: {num_interactions}")


@ray.remote
def convert_reward_to_logger_datum(rewards):
    return LogScalar("mean_reward", np.mean(rewards))


# %%
param_opt = ParameterOptimizer.from_config(config, remote=True)
replay_buffer = ReplayBuffer.from_config(config, remote=True)


workers_env = make_rollout_workers(
    name="env_worker",
    num_workers=config.num_env_workers,
    num_universes_per_worker=config.num_universes_per_env_worker,
    model=param_opt.get_model.remote(),
    params_and_state=param_opt.get_params_and_state.remote(),
    laws_factory=lambda: [
        OpenSpielEnvLaw(make_env(config.env), num_players=1),
        FrameStacker(num_frames=config.num_stacked_frames, player=0),
        make_policy_feed,
        Planner(
            num_simulations=config.num_env_simulations,
            known_bound_min=config.known_bound_min,
            known_bound_max=config.known_bound_max,
            include_tree=False,
        ),
        ActionSamplerLaw(),
        TrajectoryOutputWriter(),
    ],
)

workers_test = make_rollout_workers(
    name="test",
    num_workers=1,
    num_universes_per_worker=1,
    model=param_opt.get_model.remote(),
    params_and_state=param_opt.get_params_and_state.remote(),
    laws_factory=lambda: [
        OpenSpielEnvLaw(make_env(config.env), num_players=1),
        FrameStacker(num_frames=config.num_stacked_frames, player=0),
        make_policy_feed,
        Planner(
            num_simulations=config.num_test_simulations,
            known_bound_min=config.known_bound_min,
            known_bound_max=config.known_bound_max,
            include_tree=False,
        ),
        ActionSamplerLaw(temperature=0.1),
        output_last_step_reward,
    ],
)

workers_reanalyze = make_rollout_workers(
    name="reanalyze",
    num_workers=config.num_reanalyze_workers,
    num_universes_per_worker=config.num_universes_per_reanalyze_worker,
    model=param_opt.get_model.remote(),
    params_and_state=param_opt.get_params_and_state.remote(),
    laws_factory=lambda: [
        exit_if_no_input,
        ReanalyzeEnvLaw(),
        FrameStacker(num_frames=config.num_stacked_frames, player=0),
        make_policy_feed,
        Planner(
            num_simulations=config.num_env_simulations,
            known_bound_min=config.known_bound_min,
            known_bound_max=config.known_bound_max,
            include_tree=False,
        ),
        ActionSamplerLaw(),
        TrajectoryOutputWriter(),
    ],
)


jaxboard_logger = JAXBoardLoggerActor.remote("jaxboard_logger")
terminal_logger = TerminalLoggerActor.remote("terminal_logger")
terminal_logger.write.remote(param_opt.get_properties.remote())

# %%
with WallTimer():
    for epoch in range(config.num_epochs):
        logging.info(f"Epochs: {epoch + 1} / {config.num_epochs}")
        samples = [w.run.remote(config.num_ticks_per_epoch) for w in workers_env] + [
            w.run.remote(None) for w in workers_reanalyze
        ]

        samples_added = [replay_buffer.add_samples.remote(s) for s in samples]
        while samples_added:
            _, samples_added = ray.wait(samples_added)
            big_batch = replay_buffer.get_train_batch.remote(config.big_batch_size)
            param_opt.update.remote(
                big_batch, config.batch_size, config.num_updates_per_samples_added
            )
            for w in workers_env + workers_test + workers_reanalyze:
                w.set_params_and_state.remote(param_opt.get_params_and_state.remote())

            for w in workers_reanalyze:
                reanalyze_input = replay_buffer.get_traj_batch.remote(
                    config.num_trajs_per_reanalyze_universe
                    * config.num_universes_per_reanalyze_worker
                )
                w.set_inputs.remote(reanalyze_input)

        param_opt.log.remote()
        done = jaxboard_logger.write.remote(replay_buffer.get_stats.remote())
        # done = jaxboard_logger.write.remote(eval_result_logger_datum)
        # done = terminal_logger.write.remote(eval_result_logger_datum)

    ray.get(done)
