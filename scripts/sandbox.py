# %%
from typing import List
from time import sleep
from moozi.core.types import FragmentSample, StepSample
from moozi.core.types import TrainTarget
from moozi.nn.training import Trainer
from moozi.core.utils import stack_sequence_fields
from moozi.parameter_optimizer import ParameterServer
from dataclasses import dataclass
import random

import chex
import dm_env
import envpool
import jax
import jax.numpy as jnp
import numpy as np
from acme.wrappers.base import EnvironmentWrapper
from dotenv import load_dotenv
from flax import struct
from moozi.core import TrajectoryCollector
from moozi.core.env import (
    ArraySpec,
    GIIEnv,
    GIIEnvFeed,
    GIIEnvOut,
    GIIEnvSpec,
    GIIEnvStepFn,
    GIIVecEnv,
    OpenSpielTransformFrameWrapper,
)
from moozi.core.history_stacker import HistoryStacker
from moozi.core.scalar_transform import ScalarTransform, make_scalar_transform
from moozi.core.vis import BreakthroughVisualizer, save_gif
from moozi.driver import ConfigFactory, Driver, get_config
from moozi.gii import GII
from moozi.nn import make_model, ResNetV2Architecture, ResNetV2Spec
from moozi.planner import Planner
from moozi.replay import ReplayBuffer
from moozi.training_worker import TrainingWorker

random_key = jax.random.PRNGKey(0)

load_dotenv("/home/zeyi/moozi/.env", verbose=True)
config = get_config("/home/zeyi/moozi/examples/config.yml")
factory = ConfigFactory(config)

# %%
class EnvPoolTransformFrameWrapper(EnvironmentWrapper):
    def __init__(self, environment: dm_env.Environment, transform_fn):
        super().__init__(environment)
        self._transform_fn = transform_fn

    def _convert_timestep(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
        new_frame = timestep.observation.obs
        assert len(new_frame.shape) == 4
        new_frame = np.moveaxis(new_frame, 1, -1)
        new_obs = timestep.observation._replace(obs=new_frame)
        return timestep._replace(observation=new_obs)

    def step(self, action) -> dm_env.TimeStep:
        return self._convert_timestep(self._environment.step(action))

    def reset(self) -> dm_env.TimeStep:
        return self._convert_timestep(self._environment.reset())

    def observation_spec(self):
        obs_spec = self._environment.observation_spec()
        dummy_obs = obs_spec.obs.generate_value()
        transformed_shape = self._transform_fn(dummy_obs).shape
        return obs_spec._replace(obs=obs_spec.obs.replace(shape=transformed_shape))


@dataclass
class EnvPoolStepFn:
    dim_action: int
    num_envs: int

    def __call__(self, env, action) -> GIIEnvOut:
        ts = env.step(action)
        env_legal_actions = np.ones((self.num_envs, self.dim_action))
        legal_actions = np.c_[np.zeros(self.num_envs), env_legal_actions]

        return GIIEnvOut(
            frame=ts.observation.obs,
            is_first=ts.first(),
            is_last=ts.last(),
            to_play=np.zeros_like(ts.reward, dtype=np.int32),
            reward=ts.reward,
            legal_actions=legal_actions,
        ).cast()


class GIIPoolEnv(struct.PyTreeNode):
    name: str
    num_envs: int
    spec: GIIEnvSpec
    backend: dm_env.Environment

    @staticmethod
    def new(env_name: str, num_envs: int):
        assert num_envs >= 1
        env = envpool.make(env_name, env_type="dm", num_envs=num_envs, stack_num=1)
        env = EnvPoolTransformFrameWrapper(env, lambda x: np.moveaxis(x, 0, -1))
        dm_action_spec = env.action_spec()
        dm_obs_spec = env.observation_spec()

        # dim_action = env.spec.action_spec().num_values
        # num_envs = env.backend.spec.config.num_envs
        dim_action = dm_action_spec.num_values
        spec = GIIEnvSpec(
            num_players=1,
            dim_action=dim_action,
            frame=ArraySpec(shape=dm_obs_spec.obs.shape, dtype=np.float16),
            is_first=ArraySpec(shape=(num_envs,), dtype=np.bool8),
            is_last=ArraySpec(shape=(num_envs,), dtype=np.bool8),
            to_play=ArraySpec(shape=(num_envs,), dtype=np.int32),
            reward=ArraySpec(shape=(num_envs,), dtype=np.float16),
            legal_actions=ArraySpec(shape=(num_envs,), dtype=np.int32),
            step_fn=EnvPoolStepFn(num_envs=num_envs, dim_action=dim_action),
        )

        return GIIPoolEnv(name=env_name, num_envs=num_envs, spec=spec, backend=env)

    def init_action(self) -> chex.ArrayNumpy:
        return np.zeros(self.num_envs, dtype=np.int32)

    def init(self) -> GIIEnvFeed:
        return GIIEnvFeed(action=np.zeros(self.num_envs, dtype=np.int32))

    def step(self, feed: GIIEnvFeed) -> GIIEnvOut:
        return self.spec.step_fn(self.backend, feed)


# %%
num_envs = 12
env = GIIPoolEnv.new("Pong-v5", num_envs)
action = env.init_action()
env_out = env.step(action)

# %%
model = make_model(
    ResNetV2Architecture,
    ResNetV2Spec(
        dim_action=7,
        num_players=1,
        history_length=4,
        frame_rows=84,
        frame_cols=84,
        frame_channels=1,
        repr_rows=5,
        repr_cols=5,
        repr_channels=32,
        scalar_transform=ScalarTransform.new(-30, 30),
        repr_tower_blocks=6,
        pred_tower_blocks=2,
        dyna_tower_blocks=2,
        dyna_state_blocks=2,
    ),
)

trainer = Trainer.new(model, num_unroll_steps=5)

# %%
training_state = trainer.init(random_key)

gii = GII(
    env=env,
    stacker=HistoryStacker(
        num_rows=84,
        num_cols=84,
        num_channels=1,
        history_length=4,
        dim_action=7,
    ),
    planner=Planner(batch_size=num_envs, model=model, dim_action=7, num_players=1),
    params=training_state.params,
    state=training_state.state,
    random_key=factory.make_random_key(),
)

c = TrajectoryCollector(num_envs)
rb = ReplayBuffer(num_unroll_steps=5, num_td_steps=10, history_length=4)

# %%
from moozi.utils import WallTimer

sss = []
for i in range(1000):
    ss = gii.tick()
    sss.append(ss)

# %%
dim = 10
r = (np.random.rand(dim) // 0.4).astype(float)
v = (np.round(np.random.rand(dim), 1)).astype(float)
# b_first = np.zeros(dim, dtype=bool)
b_last = np.zeros(dim, dtype=bool)
b_last[[4]] = 1
print(r, "\n", b_last)

# %%
fragger = FragmentSample.from_step_samples(sss[:50])

# %%
from jax.experimental import host_callback as hcb
from moozi.core.utils import fifo_prepend
from flax import struct


class FragmentProcessor(struct.PyTreeNode):
    history_length: int
    num_unroll_steps: int
    discount: float
    td_step: int

    def compute_n_step_return(
        self,
        last_reward,
        is_last,
        root_value,
    ):
        # discount factor constants
        coefs = np.power(
            np.full(self.td_step, fill_value=self.discount),
            np.arange(self.td_step),
        )

        def body_fn(carry, args):
            r, b_last, v = args
            r_queue, v_queue = carry

            r_queue *= ~b_last
            v_queue *= ~b_last

            g = (coefs @ r_queue) + v_queue[-1]

            r_queue = fifo_prepend(r_queue, r)
            v_queue = fifo_prepend(v_queue, v)
            v_queue *= self.discount

            return ((r_queue, v_queue), g)

        v_queue = jnp.full(self.td_step, fill_value=root_value[-1])
        r_queue = jnp.zeros_like(v_queue)
        _, n_step_return = jax.lax.scan(
            body_fn,
            init=(r_queue, v_queue),
            xs=(last_reward, is_last, root_value),
            reverse=True,
        )
        return n_step_return

    def make_training_target(self, frag: FragmentSample):
        self._check_shape(frag)
        target = self._init_target(frag)
        print(jax.tree_map(np.shape, target))

        n_step_return = jax.vmap(self.compute_n_step_return)(
            last_reward=frag.last_reward,
            is_last=frag.is_last,
            root_value=frag.root_value,
        )

        def body_fn(i, x):
            x.append(i)
            return x

        return jax.lax.fori_loop(
            lower=self.history_length - 1,
            upper=frag.frame.shape,
            body_fun=body_fn,
            init_val=(frag, target),
        )

    def _check_shape(self, frag: FragmentSample):
        L = self.history_length
        K = self.num_unroll_steps
        B, N, H, W, C = frag.frame.shape
        chex.assert_shape(frag.frame, (B, N, H, W, C))
        chex.assert_shape(frag.action, (B, N))

    def _init_target(self, frag: FragmentSample):
        L = self.history_length
        K = self.num_unroll_steps
        B, N, H, W, C = frag.frame.shape
        _, _, A = frag.action_probs.shape
        num_targets = N - L - K
        assert num_targets >= 1
        return TrainTarget(
            frame=jnp.zeros((num_targets, L + K, H, W, C), dtype=jnp.float16),
            action=jnp.zeros((num_targets, L + K + 1), dtype=jnp.int32),
            n_step_return=jnp.zeros((num_targets, K + 1), dtype=jnp.float16),
            action_probs=jnp.zeros((num_targets, K + 1, A), jnp.float16),
            last_reward=jnp.zeros((num_targets, K + 1), jnp.float16),
            root_value=jnp.zeros((num_targets, K + 1), jnp.float16),
            to_play=jnp.zeros((num_targets,), dtype=jnp.int32),
            importance_sampling_ratio=jnp.ones((1,), dtype=jnp.float16),
        )


fragger = FragmentProcessor(
    history_length=5,
    num_unroll_steps=5,
    discount=1.0,
    td_step=10,
)
result = jax.jit(fragger.make_training_target)(frag)


# %%
result

# %%
from moozi.utils import WallTimer

t1 = WallTimer()
t2 = WallTimer()
for epoch in range(100):
    sss = []
    for i in range(1000):
        t1.start()
        ss = gii.tick()
        t1.end()
        sss.append(ss)
    t2.start()
    c.add_step_samples(sss)
    t2.end()
    trajs = c.flush()
    print(len(trajs))
    if trajs:
        avr_return = np.mean([np.sum(t.last_reward) for t in trajs])
        print(avr_return)
        rb.add_trajs(trajs)
        batch = rb.get_train_targets_batch(1024)
        training_state = trainer.sgd(training_state, batch)
        gii.params = training_state.target_params
        gii.state = training_state.target_state

# # %%
# import ray
# from moozi.utils import WallTimer
# import numpy as np
# import jax
# import jax.numpy as jnp
# import jax.profiler

# with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):

#     @ray.remote(num_gpus=1)
#     @jax.profiler.annotate_function
#     def produce():
#         x = jnp.ones(100000000)
#         return jax.device_put(x, jax.devices('gpu')[0])

#     @ray.remote(num_gpus=1)
#     @jax.profiler.annotate_function
#     def consume(x):
#         return x ** 2


#     y = ray.get(consume.remote(produce.remote()))
#     for i in range(100):
#         y = y ** 3
#     z = np.array(y)

# %%
y.device()
# %%

# %%
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# GRAVEYARD
# GRAVEYARD
# GRAVEYARD
# GRAVEYARD
# GRAVEYARD
# GRAVEYARD
# GRAVEYARD
# GRAVEYARD
# GRAVEYARD
# GRAVEYARD


# %%

# %%

# %%

# %%

# # %%
num_actions = 100
legal_actions = jnp.zeros(num_actions).at[:13].set(1)
# legal_actions = jnp.zeros(num_actions)
# legal_actions = legal_actions.at[:10].set(1)
dirichlet_alpha = 0.5
batch_size = 1
alpha = jnp.where(
    jnp.logical_not(legal_actions),
    0,
    jnp.full([num_actions], fill_value=dirichlet_alpha),
)
# alpha = jnp.full([num_actions], fill_value=dirichlet_alpha)
noise = jax.random.dirichlet(random_key, alpha=alpha, shape=(batch_size,))
random_key, _ = jax.random.split(random_key)
np.round(jnp.where(legal_actions, noise * 0.25, 0), 3)

# %%
graph = pygraphviz.AGraph(directed=True, imagepath=image_dir, dpi=72)
graph.node_attr.update(
    imagescale=True,
    shape="box",
    imagepos="tc",
    fixed_size=True,
    labelloc="b",
    fontname="Courier New",
)
graph.edge_attr.update(fontname="Courier New")

graph.add_node(0, label="some label", width=2.5, height=4, image="image_path")
# Add all other nodes and connect them up.
for i, node in enumerate(nodes):
    graph.add_node(
        child_id, label="some other label", width=2.5, height=4, image="image_path"
    )
