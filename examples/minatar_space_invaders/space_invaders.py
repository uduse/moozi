# %%
import uuid
from PIL import Image, ImageDraw, ImageFont
import seaborn as sns
import tree
import contextlib
from functools import partial
import chex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Set, Tuple, Union

import jax

import moozi
import numpy as np
import ray
from acme.utils.tree_utils import stack_sequence_fields
from dotenv import load_dotenv
from loguru import logger
import moozi as mz
from moozi.core import scalar_transform
from moozi.core.env import make_env
from moozi.core.scalar_transform import ScalarTransform, make_scalar_transform
from moozi.core.tape import make_tape, include, exclude
from moozi.nn.nn import NNModel, make_model
from moozi.nn.training import make_training_suite
from moozi.parameter_optimizer import ParameterOptimizer, ParameterServer
from moozi.replay import ReplayBuffer
from moozi.core.link import link
from moozi.laws import *
from moozi.planner import make_planner

# from moozi.rollout_worker import RolloutWorkerWithWeights, make_rollout_workers
from moozi.utils import WallTimer
from omegaconf import OmegaConf

load_dotenv()

logger.remove()
logger.add(sys.stderr, level="INFO")

config = OmegaConf.load(Path(__file__).parent / "config.yml")
OmegaConf.resolve(config)
print(OmegaConf.to_yaml(config))


class Universe:
    def __init__(self, tape, law: Law) -> None:
        assert isinstance(tape, dict)
        self.tape = tape
        self.law = law

    def tick(self):
        self.tape = self.law.apply(self.tape)

    def run(self):
        while True:
            self.tick()
            if self.tape["quit"]:
                break
        return self.flush()

    def flush(self):
        ret = self.tape["output_buffer"]
        logger.debug(f"flushing {len(ret)} trajectories")
        self.tape["output_buffer"] = tuple()
        return ret


# %%
def make_min_atar_gif_recorder(n_channels=6):
    cmap = sns.color_palette("cubehelix", n_channels)
    cmap.insert(0, (0, 0, 0))
    cs = np.array([cmap[i] for i in range(n_channels + 1)])

    def malloc():
        Path("gifs").mkdir(parents=True, exist_ok=True)
        return {"images": []}

    def apply(is_last, obs, images: List[Image.Image], root_value):
        numerical_state = np.array(
            np.amax(obs[0] * np.reshape(np.arange(n_channels) + 1, (1, 1, -1)), 2)
            + 0.5,
            dtype=int,
        )
        rgbs = np.array(cs[numerical_state - 1] * 255, dtype=np.uint8)
        img = Image.fromarray(rgbs)
        img = img.resize((img.width * 40, img.height * 40), Image.NEAREST)
        draw = ImageDraw.Draw(img)
        content = f"v = {float(root_value[0]):.2f}"
        font = ImageFont.truetype("courier.ttf", 14)
        draw.text((0, 0), content, fill="black", font=font)
        images = images + [img]
        if is_last[0] and images:
            counter = 0
            while gif_fpath := (Path("gifs") / f"{counter}.gif"):
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


# %%
class RolloutWorker:
    def __init__(
        self, universe_factory: Callable, name: str = "rollout_worker"
    ) -> None:
        self.universe = universe_factory()
        self.name = name

        from loguru import logger

        logger.remove()
        logger.add(f"logs/rollout_worker.{self.name}.debug.log", level="DEBUG")
        logger.add(f"logs/rollout_worker.{self.name}.info.log", level="INFO")
        logger.info(
            f"RolloutWorker created, name: {self.name}, universe include {self.universe.tape.keys()}"
        )

    def run(self):
        return self.universe.run()

    def set(self, key, value):
        if isinstance(value, ray.ObjectRef):
            value = ray.get(value)
        self.universe.tape[key] = value


# %%
scalar_transform = make_scalar_transform(**config.scalar_transform)
nn_arch_cls = eval(config.nn.arch_cls)
nn_spec = eval(config.nn.spec_cls)(
    **config.nn.spec_kwargs, scalar_transform=scalar_transform
)
gif_recorder = make_min_atar_gif_recorder(n_channels=6)
model = make_model(nn_arch_cls, nn_spec)

# %%
def make_env_worker_universe(config):
    num_envs = config.train.env_workers.num_envs
    vec_env = make_vec_env(config.env.name, num_envs)
    frame_stacker = make_batch_frame_stacker(
        num_envs,
        config.env.num_rows,
        config.env.num_cols,
        config.env.num_channels,
        config.num_stacked_frames,
    )
    planner = make_planner(
        num_envs=num_envs,
        dim_actions=config.dim_actions,
        model=model,
        num_simulations=10,
        dirichlet_fraction=config.mcts.dirichlet_fraction,
        dirichlet_alpha=config.mcts.dirichlet_alpha,
        temperature=config.mcts.temperature,
    )
    policy = sequential([frame_stacker, planner])
    policy.apply = chex.assert_max_traces(n=1)(policy.apply)
    policy.apply = jax.jit(policy.apply, backend="gpu")

    traj_writer = make_traj_writer(num_envs)
    terminator = make_terminator(num_envs)

    final_law = sequential(
        [
            vec_env,
            gif_recorder,
            policy,
            traj_writer,
            terminator,
        ]
    )
    tape = make_tape(seed=config.seed)
    tape.update(final_law.malloc())
    return Universe(tape, final_law)


def make_test_worker_universe(config):
    vec_env = make_vec_env(config.env.name, 1)
    frame_stacker = make_batch_frame_stacker(
        1,
        config.env.num_rows,
        config.env.num_cols,
        config.env.num_channels,
        config.num_stacked_frames,
    )
    planner = make_planner(
        num_envs=1,
        dim_actions=config.dim_actions,
        model=model,
        **config.train.test_worker.planner
    )
    policy = sequential([frame_stacker, planner])
    policy.apply = chex.assert_max_traces(n=1)(policy.apply)
    policy.apply = jax.jit(policy.apply, backend="gpu")

    traj_writer = make_traj_writer(1)
    terminator = make_terminator(size=10)

    final_law = sequential(
        [
            vec_env,
            gif_recorder,
            policy,
            traj_writer,
            terminator,
        ]
    )
    tape = make_tape(seed=config.seed)
    tape.update(final_law.malloc())
    return Universe(tape, final_law)


# %%
def training_suite_factory(config):
    return partial(
        make_training_suite,
        seed=config.seed,
        nn_arch_cls=nn_arch_cls,
        nn_spec=nn_spec,
        weight_decay=config.train.weight_decay,
        lr=config.train.lr,
        num_unroll_steps=config.num_unroll_steps,
    )


# %%
ps = ray.remote(num_gpus=config.param_opt.num_gpus)(ParameterServer).remote(
    training_suite_factory=training_suite_factory(config), use_remote=True
)

# %%
train_workers = [
    ray.remote(num_gpus=config.train.env_workers.num_gpus)(RolloutWorker).remote(
        partial(make_env_worker_universe, config), name=f"rollout_worker_{i}"
    )
    for i in range(config.train.env_workers.num_workers)
]

# %%
for w in train_workers:
    w.set.remote("params", ps.get_params.remote())
    w.set.remote("state", ps.get_state.remote())

# %%
rb = ray.remote(ReplayBuffer).remote(**config.replay)

# %%
ps.get_properties.remote()
for _ in range(config.train.num_epochs):
    train_targets = []
    for w in train_workers:
        sample = w.run.remote()
        train_targets.append(rb.add_trajs.remote(sample))

    # sync
    ray.get(train_targets)

    batch = rb.get_train_targets_batch.remote(batch_size=config.train.batch_size)
    # rb.get_stats.remote()
    ps.update.remote(batch, batch_size=config.train.batch_size)
    for w in train_workers:
        w.set.remote("params", ps.get_params.remote())
        w.set.remote("state", ps.get_state.remote())
    ray.get(ps.log_tensorboard.remote())

# with WallTimer():
#     for epoch in range(config.num_epochs):
#         for w in workers_env + workers_test + workers_reanalyze:
#             w.set_params_and_state.remote(param_opt.get_params_and_state.remote())
#         logger.debug(f"Get params and state scheduled, {len(traj_futures)=}")

#         while traj_futures:
#             traj, traj_futures = ray.wait(traj_futures)
#             traj = traj[0]
#             replay_buffer.add_trajs.remote(traj)

#         if epoch >= config.epoch_train_start:
#             logger.debug(f"Add trajs scheduled, {len(traj_futures)=}")
#             train_batch = replay_buffer.get_train_targets_batch.remote(
#                 config.big_batch_size
#             )
#             logger.debug(f"Get train targets batch scheduled, {len(traj_futures)=}")
#             update_done = param_opt.update.remote(train_batch, config.batch_size)
#             logger.debug(f"Update scheduled")

#         jaxboard_logger.write.remote(replay_buffer.get_stats.remote())
#         env_trajs = [w.run.remote(config.num_ticks_per_epoch) for w in workers_env]
#         reanalyze_trajs = [w.run.remote(None) for w in workers_reanalyze]
#         traj_futures = env_trajs + reanalyze_trajs

#         if epoch % config.test_interval == 0:
#             test_result = workers_test[0].run.remote(6 * 20)
#             test_result_datum = convert_test_result_record.remote(test_result)
#         logger.debug(f"test result scheduled.")

#         for w in workers_reanalyze:
#             reanalyze_input = replay_buffer.get_train_targets_batch.remote(
#                 config.num_trajs_per_reanalyze_universe
#                 * config.num_universes_per_reanalyze_worker
#             )
#             w.set_inputs.remote(reanalyze_input)
#         logger.debug(f"reanalyze scheduled.")
#         test_done = jaxboard_logger.write.remote(test_result_datum)

#         param_opt.log.remote()
#         logger.info(f"Epochs: {epoch + 1} / {config.num_epochs}")

# logger.debug(ray.get(test_done))
# ray.get(jaxboard_logger.close.remote())
# ray.get(param_opt.close.remote())
