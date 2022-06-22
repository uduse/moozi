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
from moozi.core.env import make_env, VecEnv
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
    def __init__(self, tape, law) -> None:
        assert isinstance(tape, dict)
        self.tape = tape
        self.law = law

    def tick(self):
        self.tape = self.law(self.tape)

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

    def forward(is_last, obs, images: List[Image.Image], root_value):
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
            logger.info("gif saved to ", str(gif_fpath))
            images = []
        return {"images": images}

    return malloc, forward


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

    def set_universe_property(self, key, value):
        if isinstance(value, ray.ObjectRef):
            value = ray.get(value)
        self.universe.tape[key] = value


@dataclass
class L:
    name: str
    apply: Callable
    keys: Set[str]



def make_universe(config):
    scalar_transform = make_scalar_transform(**config.scalar_transform)
    nn_arch_cls = eval(config.nn.arch_cls)
    nn_spec = eval(config.nn.spec_cls)(
        **config.nn.spec_kwargs, scalar_transform=scalar_transform
    )

    model = make_model(nn_arch_cls, nn_spec)
    num_envs = config.train.env_workers.num_envs
    tape = {
        "random_key": jax.random.PRNGKey(config.seed),
        "output_buffer": tuple(),
        "quit": False,
    }
    vec_env = VecEnv(config.env.name, num_envs)
    tape.update(vec_env.malloc())
    vec_env = link(vec_env)

    m, gifer = make_min_atar_gif_recorder()
    tape.update(m())
    gifer = link(gifer)

    frame_stacker = BatchFrameStacker(
        num_envs,
        config.env.num_rows,
        config.env.num_cols,
        config.env.num_channels,
        config.num_stacked_frames,
    )
    tape.update(frame_stacker.malloc())
    frame_stacker = link(frame_stacker)

    malloc, planner = make_planner(
        num_envs=num_envs,
        dim_actions=config.dim_actions,
        model=model,
        num_simulations=10,
        dirichlet_fraction=config.mcts.dirichlet_fraction,
        dirichlet_alpha=config.mcts.dirichlet_alpha,
        temperature=1.0,
    )
    tape.update(malloc())
    planner = link(planner)

    traj_writer = TrajWriter(num_envs)
    tape.update(traj_writer.malloc())
    traj_writer = link(traj_writer)

    terminator = link(make_terminator(num_envs).apply)

    @partial(jax.jit, backend="cpu")
    @chex.assert_max_traces(n=1)
    def policy(tape):
        tape = frame_stacker(tape)
        tape = planner(tape)
        return tape

    def law(tape):
        tape = vec_env(tape)
        with include(
            tape,
            {
                "obs",
                "stacked_frames",
                "params",
                "state",
                "random_key",
                "is_first",
                "is_last",
            },
        ) as tape_slice:
            updates = policy(tape_slice)
        tape.update(updates)
        tape = traj_writer(tape)
        tape = gifer(tape)
        tape = terminator(tape)
        return tape

    return Universe(tape, law)


# %%
def make_universe_v2(config):
    scalar_transform = make_scalar_transform(**config.scalar_transform)
    nn_arch_cls = eval(config.nn.arch_cls)
    nn_spec = eval(config.nn.spec_cls)(
        **config.nn.spec_kwargs, scalar_transform=scalar_transform
    )

    model = make_model(nn_arch_cls, nn_spec)

    tape = make_tape(seed=config.seed)

    num_envs = config.train.env_workers.num_envs
    vec_env = make_vec_env(config.env.name, num_envs)
    gif_recorder = make_min_atar_gif_recorder()
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
        temperature=1.0,
    )
    
    traj_writer = make_traj_writer

    terminator = link(make_terminator(num_envs).apply)

    @partial(jax.jit, backend="cpu")
    @chex.assert_max_traces(n=1)
    def policy(tape):
        tape = frame_stacker(tape)
        tape = planner(tape)
        return tape

    def law(tape):
        tape = vec_env(tape)
        with include(
            tape,
            {
                "obs",
                "stacked_frames",
                "params",
                "state",
                "random_key",
                "is_first",
                "is_last",
            },
        ) as tape_slice:
            updates = policy(tape_slice)
        tape.update(updates)
        tape = traj_writer(tape)
        tape = gifer(tape)
        tape = terminator(tape)
        return tape

    return Universe(tape, law)


# %%
def training_suite_factory(config):
    scalar_transform = make_scalar_transform(**config.scalar_transform)
    nn_arch_cls = eval(config.nn.arch_cls)
    nn_spec_cls = eval(config.nn.spec_cls)(
        **config.nn.spec_kwargs, scalar_transform=scalar_transform
    )

    return partial(
        make_training_suite,
        seed=config.seed,
        nn_arch_cls=nn_arch_cls,
        nn_spec=nn_spec_cls,
        weight_decay=config.train.weight_decay,
        lr=config.train.lr,
        num_unroll_steps=config.num_unroll_steps,
    )


# %%
parameter_server = ParameterServer(
    training_suite_factory=training_suite_factory(config),
)

# %%
worker = RolloutWorker(partial(make_universe_v2, config), name="rollout_worker")

# %%
worker.set_universe_property("params", parameter_server.get_params())
worker.set_universe_property("state", parameter_server.get_state())

# %%
replay_buffer = ReplayBuffer(**config.replay)

# %%
print(parameter_server.get_properties())
for _ in range(1000):
    result = worker.universe.run()
    replay_buffer.add_trajs(result)

    batch = replay_buffer.get_train_targets_batch(128)
    print(replay_buffer.get_stats())
    print(parameter_server.update(batch, batch_size=128))
    worker.set_universe_property("params", parameter_server.get_params())
    worker.set_universe_property("state", parameter_server.get_state())
    parameter_server.log_tensorboard()
