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

config = OmegaConf.load("/moozi/examples/minatar_space_invaders/config.yml")
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
def _termination_penalty(is_last, reward):
    reward_overwrite = jax.lax.cond(
        is_last,
        lambda: reward - 10.0,
        lambda: reward,
    )
    return {"reward": reward_overwrite}


penalty = Law.wrap(_termination_penalty)
penalty.apply = link(jax.vmap(unlink(penalty.apply)))

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

    cat_obs = Law(
        name="cat_obs",
        malloc=lambda: {
            "obs": jnp.zeros(
                (
                    num_envs,
                    config.nn.spec_kwargs.obs_rows,
                    config.nn.spec_kwargs.obs_cols,
                    config.nn.spec_kwargs.obs_channels,
                )
            )
        },
        apply=link(
            lambda stacked_frames, stacked_actions: {
                "obs": jnp.concatenate([stacked_frames, stacked_actions], axis=-1)
            }
        ),
        read={"stacked_frames", "stacked_actions"},
    )

    planner = make_planner(model=model, **config.train.env_worker.planner)

    traj_writer = make_traj_writer(num_envs)
    terminator = make_terminator(num_envs)

    final_law = sequential(
        [
            vec_env,
            penalty,
            sequential(
                [
                    frame_stacker,
                    cat_obs,
                    planner,
                ],
            ).jit(backend="gpu"),
            make_min_atar_gif_recorder(n_channels=6, root_dir="env_worker_gifs"),
            traj_writer,
            terminator,
        ]
    )
    tape = make_tape(seed=config.seed)
    tape.update(final_law.malloc())
    return Universe(tape, final_law)


# %%
def make_test_worker_universe(config):
    policy = sequential(
        [
            make_batch_stacker(
                1,
                config.env.num_rows,
                config.env.num_cols,
                config.env.num_channels,
                config.num_stacked_frames,
                config.dim_action,
            ),
            make_planner(model=model, **config.train.test_worker.planner),
        ]
    )
    policy.apply = chex.assert_max_traces(n=1)(policy.apply)
    policy.apply = jax.jit(policy.apply, backend="gpu")

    final_law = sequential(
        [
            make_vec_env(config.env.name, 1),
            policy,
            make_min_atar_gif_recorder(n_channels=6, root_dir="test_worker_gifs"),
            make_traj_writer(1),
            make_reward_terminator(5),
        ]
    )
    tape = make_tape(seed=config.seed)
    tape.update(final_law.malloc())
    return Universe(tape, final_law)


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
ps = ParameterServer(training_suite_factory=training_suite_factory(config))
rb = ReplayBuffer(**config.replay)

# %%
checkpoint = "/root/.local/share/virtualenvs/moozi-g1CZ00E9/.guild/runs/3a44c2de50bb413e9ae79fb3a8976187/checkpoints/9765.pkl"
ps.restore(checkpoint)

# %%
w = RolloutWorker(partial(make_env_worker_universe, config))

# %%
w.set("params", ps.get_params())
w.set("state", ps.get_state())
for _ in range(1):
    trajs = w.run()
    rb.add_trajs(trajs)

# %%
trajs_batch = rb.get_trajs_batch(2)

# %%
@Law.wrap
def to_step_sample(
    frame,
    reward,
    is_first,
    is_last,
    to_play,
    legal_actions_mask,
    root_value,
    action_probs,
    action,
):
    return {
        "step_sample": StepSample(
            frame=frame,
            last_reward=reward,
            is_first=is_first,
            is_last=is_last,
            to_play=to_play,
            legal_actions_mask=legal_actions_mask,
            root_value=root_value,
            action_probs=action_probs,
            action=action,
        )
    }


# %%

# re_law = make_re()
# tape = make_tape(seed=config.seed)
# tape.update(re_law.malloc())
# tape["params"] = ps.get_params()
# tape["state"] = ps.get_state()
# tape['traj'] = trajs_batch[0]

# universe = Universe(tape, re_law)
# result = universe.run()

# %%
# jnp.mean(-jnp.sum(result[0].action_probs  * jnp.log(trajs_batch[0].action_probs + 1e-15), axis=-1))

# %%
# from acme.utils.tree_utils import stack_sequence_fields
# from acme.jax.utils import squeeze_batch_dim

# new_traj = stack_sequence_fields(
#     [squeeze_batch_dim(step_sample) for step_sample in step_samples]
# )

# %%
# from moozi.nn.training import make_target_from_traj

# make_target_from_traj(
#     new_traj,
#     start_idx=0,
#     discount=0.99,
#     num_unroll_steps=5,
#     num_td_steps=1,
#     num_stacked_frames=4,
# )

# %%
# w_re = RolloutWorker(partial(make_reanalyze_universe, config))
# w_re.set("params", ps.get_params())
# w_re.set("state", ps.get_state())

# %%
# w_re.set("train_target", old)
# result = w_re.run()

# %%
# tree.map_structure(lambda x: x.shape, old)

# # %%
# tape["policy_cross_entropy"]

# # %%
# def output_ones():
#     return jnp.ones((10, 2))


# jax.vmap(output_ones, axis_size=10)()
# # %%
