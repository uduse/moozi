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
from moozi.logging import JAXBoardLoggerRemote, TerminalLoggerRemote
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
print(OmegaConf.to_yaml(config, resolve=True))


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

    def apply(
        is_last,
        obs,
        images: List[Image.Image],
        root_value,
        q_values,
        action_probs,
        action,
    ):
        numerical_state = np.array(
            np.amax(obs[0] * np.reshape(np.arange(n_channels) + 1, (1, 1, -1)), 2)
            + 0.5,
            dtype=int,
        )
        rgbs = np.array(cs[numerical_state - 1] * 255, dtype=np.uint8)
        img = Image.fromarray(rgbs)
        img = img.resize((img.width * 40, img.height * 40), Image.NEAREST)
        draw = ImageDraw.Draw(img)
        content = (
            f"A {str(action)}\n"
            f"V {str(np.round(root_value[0], 2))}\n"
            f"π {str(np.round(action_probs[0], 2))}\n"
            f"Q {str(np.round(q_values[0], 2))}\n"
        )
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
model = make_model(nn_arch_cls, nn_spec)

# %%
def make_env_worker_universe(config):
    num_envs = config.train.env_worker.num_envs
    vec_env = make_vec_env(config.env.name, num_envs)
    frame_stacker = make_batch_stacker(
        num_envs,
        config.env.num_rows,
        config.env.num_cols,
        config.env.num_channels,
        config.num_stacked_frames,
        config.dim_action,
    )
    planner = make_planner(model=model, **config.train.env_worker.planner)
    policy = sequential([frame_stacker, planner])
    policy.apply = chex.assert_max_traces(n=1)(policy.apply)
    policy.apply = jax.jit(policy.apply, backend="gpu")

    traj_writer = make_traj_writer(num_envs)
    terminator = make_terminator(num_envs)

    final_law = sequential(
        [
            vec_env,
            policy,
            traj_writer,
            terminator,
        ]
    )
    tape = make_tape(seed=config.seed)
    tape.update(final_law.malloc())
    return Universe(tape, final_law)


# %%
def make_test_worker_universe(config):
    vec_env = make_vec_env(config.env.name, 1)
    frame_stacker = make_batch_stacker(
        1,
        config.env.num_rows,
        config.env.num_cols,
        config.env.num_channels,
        config.num_stacked_frames,
    )
    planner = make_planner(model=model, **config.train.test_worker.planner)
    policy = sequential([frame_stacker, planner])
    policy.apply = chex.assert_max_traces(n=1)(policy.apply)
    policy.apply = jax.jit(policy.apply, backend="gpu")
    gif_recorder = make_min_atar_gif_recorder(n_channels=6)

    final_law = sequential(
        [
            vec_env,
            policy,
            gif_recorder,
            make_traj_writer(1),
            make_reward_terminator(5),
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
    ray.remote(num_gpus=config.train.env_worker.num_gpus)(RolloutWorker).remote(
        partial(make_env_worker_universe, config), name=f"rollout_worker_{i}"
    )
    for i in range(config.train.env_worker.num_workers)
]

# %%
test_worker = ray.remote(num_gpus=config.train.test_worker.num_gpus)(
    RolloutWorker
).remote(partial(make_test_worker_universe, config), name="test_worker")

rb = ray.remote(ReplayBuffer).remote(**config.replay)

jb_logger = JAXBoardLoggerRemote.remote()
terminal_logger = TerminalLoggerRemote.remote()

for epoch in range(config.train.num_epochs):

    for w in train_workers:
        w.set.remote("params", ps.get_params.remote())
        w.set.remote("state", ps.get_state.remote())

    if epoch % config.train.test_worker.interval == 0:
        test_worker.set.remote("params", ps.get_params.remote())
        test_worker.set.remote("state", ps.get_state.remote())

    # launch test
    test_result = test_worker.run.remote()
    terminal_logger.write.remote(test_result)
    jb_logger.write.remote(test_result)

    # generate train targets
    train_targets = []
    for w in train_workers:
        sample = w.run.remote()
        train_targets.append(rb.add_trajs.remote(sample))

    # sync
    ray.get(train_targets)

    batch = rb.get_train_targets_batch.remote(batch_size=config.train.batch_size)
    ps.update.remote(batch, batch_size=config.train.batch_size)
    if epoch % config.param_opt.save_interval == 0:
        ps.save.remote()
    ray.get(ps.log_tensorboard.remote())
