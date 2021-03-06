# %%
import sys
import pprint
import copy
from dataclasses import dataclass, field
import os
from typing import List, Tuple

import moozi as mz
import numpy as np
import ray
from loguru import logger
from moozi.core import Config, link
from moozi.core.env import make_env
from moozi.laws import (
    FrameStacker,
    MinAtarEnvLaw,
    ReanalyzeEnvLawV2,
    TrajectoryOutputWriter,
    exit_if_no_input,
    increment_tick,
    make_policy_feed,
)
from moozi.logging import (
    JAXBoardLoggerRemote,
    LogScalar,
    LogText,
    TerminalLoggerRemote,
)
from moozi.mcts import ActionSamplerLaw, Planner
from moozi.parameter_optimizer import ParameterOptimizer
from moozi.replay import ReplayBuffer
from moozi.rollout_worker import make_rollout_workers
from moozi.utils import WallTimer
from dotenv import load_dotenv

load_dotenv()

# %%
logger.remove()
logger.add(sys.stderr, level="INFO")
# logger.add("logs/main.log", level="DEBUG")

# %%
seed = 0
num_epochs = 100
lr = 5e-3

config = Config()
config.env = f"MinAtar:Breakout-v1"
config.known_bound_min = 0
config.known_bound_max = 1

config.discount = 0.99
config.num_unroll_steps = 2
config.num_td_steps = 100
config.num_stacked_frames = 1
config.lr = lr

config.replay_max_size = 100000

config.num_epochs = num_epochs
config.epoch_train_start = 2

num_batches_per_epoch = 64
config.batch_size = 256
config.big_batch_size = config.batch_size * num_batches_per_epoch

config.num_env_workers = 6
config.num_ticks_per_epoch = 30
config.num_universes_per_env_worker = 50

reanalyze_workers = 0
config.num_reanalyze_workers = reanalyze_workers
config.num_universes_per_reanalyze_worker = 20
config.num_trajs_per_reanalyze_universe = 2

config.weight_decay = 5e-2
config.nn_arch_cls = mz.nn.ResNetArchitecture

config.test_interval = 5

game_num_rows = 10
game_num_cols = 10
env_spec = mz.make_spec(config.env)
single_frame_shape = env_spec.observations.shape
obs_channels = single_frame_shape[-1] * config.num_stacked_frames
repr_channels = 6
dim_action = env_spec.actions.num_values
config.nn_spec = mz.nn.ResNetSpec(
    obs_rows=game_num_rows,
    obs_cols=game_num_cols,
    obs_channels=obs_channels,
    repr_rows=game_num_rows,
    repr_cols=game_num_cols,
    repr_channels=repr_channels,
    dim_action=dim_action,
    repr_tower_blocks=6,
    repr_tower_dim=6,
    pred_tower_blocks=6,
    pred_tower_dim=6,
    dyna_tower_blocks=6,
    dyna_tower_dim=6,
    dyna_state_blocks=6,
)

logger.info(f"config: {pprint.pformat(config.asdict())}")

# %%
num_interactions = (
    config.num_epochs
    * config.num_ticks_per_epoch
    * config.num_env_workers
    * config.num_universes_per_env_worker
)
logger.info(f"num_interactions: {num_interactions}")

# %%
total_update_calls = config.num_epochs * (
    config.num_env_workers + config.num_reanalyze_workers
)
total_mini_batchs = total_update_calls * int(config.big_batch_size / config.batch_size)
total_samples = total_update_calls * config.big_batch_size
logger.info(f"total_update_calls: {total_update_calls}")
logger.info(f"total_mini_batchs: {total_mini_batchs}")
logger.info(f"total_samples: {total_samples}")


# %%
def frame_to_str(frame):
    frame = frame[0]
    items = []
    for irow, row in enumerate(frame):
        for val in row:
            if np.isclose(val, 0.0):
                items.append(".")
                continue
            assert np.isclose(val, 1), val
            if irow == len(frame) - 1:
                items.append("X")
            else:
                items.append("O")
        items.append("\n\n")
    return "".join(items)


# %%
param_opt = ParameterOptimizer.from_config(config, remote=True)
replay_buffer = ReplayBuffer.make(config, remote=True)


workers_env = make_rollout_workers(
    name="env_worker",
    num_workers=config.num_env_workers,
    num_universes_per_worker=config.num_universes_per_env_worker,
    model=param_opt.get_model.remote(),
    params_and_state=param_opt.get_params_and_state.remote(),
    laws_factory=lambda: [
        MinAtarEnvLaw(make_env(config.env)),
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
        increment_tick,
    ],
    num_gpus=0.2,
)


@link
@dataclass
class TestResultOutputWriter:
    renderings: List[str] = field(default_factory=list)
    action_probs: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    reward: float = 0.0

    def __call__(
        self, obs, action_probs, action, reward, is_first, is_last, output_buffer
    ):
        if is_first:
            self.renderings = []
            self.action_probs = []
            self.actions = []

        # self.renderings.append(frame_to_str(obs))
        self.action_probs.append(np.round(action_probs, 2))
        self.actions.append(action)

        update = {}

        if is_last:
            self.reward = reward
            update["output_buffer"] = output_buffer + (copy.deepcopy(self),)

        return update

    # def __str__(self):
    #     s = ""
    #     for rendering, action_probs, action in zip(
    #         self.renderings, self.action_probs, self.actions
    #     ):
    #         s += f"{rendering}\n\n"
    #         s += f"{action_probs} -> {action}\n\n"
    #     return s


# %%
@ray.remote
def convert_test_result_record(recorders: Tuple[TestResultOutputWriter, ...]):
    return [
        LogScalar("mean_reward", float(np.mean([x.reward for x in recorders]))),
        # LogText("test_vis", "\n\n\n\n".join([str(x) for x in recorders])),
    ]


workers_test = make_rollout_workers(
    name="test",
    num_workers=1,
    num_universes_per_worker=1,
    model=param_opt.get_model.remote(),
    params_and_state=param_opt.get_params_and_state.remote(),
    laws_factory=lambda: [
        MinAtarEnvLaw(make_env(config.env)),
        FrameStacker(num_frames=config.num_stacked_frames, player=0),
        make_policy_feed,
        Planner(
            num_simulations=config.num_test_simulations,
            known_bound_min=config.known_bound_min,
            known_bound_max=config.known_bound_max,
            include_tree=True,
        ),
        ActionSamplerLaw(temperature=0.2),
        TestResultOutputWriter(),
        increment_tick,
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
        ReanalyzeEnvLawV2(),
        # FrameStacker(num_frames=config.num_stacked_frames, player=0),
        make_policy_feed,
        Planner(
            num_simulations=config.num_env_simulations,
            known_bound_min=config.known_bound_min,
            known_bound_max=config.known_bound_max,
            include_tree=False,
        ),
        TrajectoryOutputWriter(),
        increment_tick,
    ],
    num_gpus=0.2,
)


jaxboard_logger = JAXBoardLoggerRemote.remote("jaxboard_logger")
terminal_logger = TerminalLoggerRemote.remote("terminal_logger")
terminal_logger.write.remote(param_opt.get_properties.remote())

# %%
traj_futures: List[ray.ObjectRef] = []

# %%
enable_test = True

# Driver
with WallTimer():
    for epoch in range(config.num_epochs):
        for w in workers_env + workers_test + workers_reanalyze:
            w.set_params_and_state.remote(param_opt.get_params_and_state.remote())
        logger.debug(f"Get params and state scheduled, {len(traj_futures)=}")

        while traj_futures:
            traj, traj_futures = ray.wait(traj_futures)
            traj = traj[0]
            replay_buffer.add_trajs.remote(traj)

        if epoch >= config.epoch_train_start:
            logger.debug(f"Add trajs scheduled, {len(traj_futures)=}")
            train_batch = replay_buffer.get_train_targets_batch.remote(
                config.big_batch_size
            )
            logger.debug(f"Get train targets batch scheduled, {len(traj_futures)=}")
            update_done = param_opt.update.remote(train_batch, config.batch_size)
            logger.debug(f"Update scheduled")

        jaxboard_logger.write.remote(replay_buffer.get_stats.remote())
        env_trajs = [w.run.remote(config.num_ticks_per_epoch) for w in workers_env]
        reanalyze_trajs = [w.run.remote(None) for w in workers_reanalyze]
        traj_futures = env_trajs + reanalyze_trajs

        if enable_test:
            if epoch % config.test_interval == 0:
                test_result = workers_test[0].run.remote(1000)
                test_result_datum = convert_test_result_record.remote(test_result)
            logger.debug(f"test result scheduled.")

        for w in workers_reanalyze:
            reanalyze_input = replay_buffer.get_train_targets_batch.remote(
                config.num_trajs_per_reanalyze_universe
                * config.num_universes_per_reanalyze_worker
            )
            w.set_inputs.remote(reanalyze_input)
        logger.debug(f"reanalyze scheduled.")

        if enable_test:
            test_done = jaxboard_logger.write.remote(test_result_datum)

        param_opt.log.remote()
        logger.info(f"Epochs: {epoch + 1} / {config.num_epochs}")

if enable_test:
    logger.debug(ray.get(test_done))
ray.get(jaxboard_logger.close.remote())
ray.get(param_opt.close.remote())
