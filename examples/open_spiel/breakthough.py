# %%
import tree
from IPython.display import display
import jax
import jax.numpy as jnp
import numpy as np
import chex
from flax import struct
import random
from moozi.core import make_env, StepSample, TrajectorySample
from moozi.core.utils import (
    stack_sequence_fields,
    unstack_sequence_fields,
    HistoryStacker,
)
from moozi.core.vis import BreakthroughVisualizer, save_gif
from moozi.core.env import GIIEnv, GIIVecEnv, GIIEnvFeed, GIIEnvOut
from moozi.nn import RootFeatures, NNModel
from moozi.planner import Planner
from lib import get_model, get_config

# %%
from typing import Dict, Tuple, Optional, List
import haiku as hk


class Agent(struct.PyTreeNode):
    num_players: int
    batch_size: int
    stacker: HistoryStacker

    class AgentFeed(struct.PyTreeNode):
        params: hk.Params
        state: hk.State
        planner: Planner
        env_out: GIIEnvOut
        last_action: chex.Array

    class AgentOut(struct.PyTreeNode):
        planner_out: Planner.PlannerOut

    class AgentState(struct.PyTreeNode):
        stacker: HistoryStacker.StackerState
        random_key: chex.PRNGKey

    def init(self, random_key: chex.PRNGKey) -> AgentState:
        return self.AgentState(
            stacker=jax.vmap(self.stacker.init, axis_size=self.batch_size)(),
            random_key=random_key,
        )

    def step(
        self,
        agent_state: "AgentState",
        agent_feed: "AgentFeed",
    ) -> Tuple["AgentState", "AgentOut"]:
        stacker_state = jax.vmap(self.stacker.apply)(
            state=agent_state.stacker,
            frame=agent_feed.env_out.frame,
            action=agent_feed.last_action,
            is_first=agent_feed.env_out.is_first,
        )
        master_key, planner_key = jax.random.split(agent_state.random_key)
        root_feats = RootFeatures(
            stacker_state.frames,
            stacker_state.actions,
            agent_feed.env_out.to_play,
        )
        planner_feed = Planner.PlannerFeed(
            params=agent_feed.params,
            state=agent_feed.state,
            root_feats=root_feats,
            legal_actions=agent_feed.env_out.legal_actions,
            random_key=planner_key,
        )
        planner_out = agent_feed.planner.run(planner_feed)

        return (
            self.AgentState(stacker=stacker_state, random_key=master_key),
            self.AgentOut(planner_out),
        )


class AgentEnvironmentInterface:
    def __init__(
        self,
        env_name: str,
        agent: Agent,
        planner: Planner,
        params: hk.Params,
        state: hk.State,
        random_key: chex.PRNGKey,
        num_envs: int = 1,
    ):
        self.env = GIIVecEnv(env_name, num_envs=num_envs)
        self.env_feed = self.env.init()
        self.agent = agent
        self.agent_state = agent.init(random_key)
        self.agent_step_fn = jax.jit(
            chex.assert_max_traces(self.agent.step, n=10), device=jax.devices("gpu")[-1]
        )

        self.params = params
        self.state = state
        self.planner = planner

    def tick(self):
        env_out = self.env.step(self.env_feed)
        self.env_feed.reset = env_out.is_last
        agent_feed = Agent.AgentFeed(
            params=self.params,
            state=self.state,
            planner=self.planner,
            env_out=env_out,
            last_action=self.env_feed.action,
        )
        self.agent_state, agent_out = self.agent_step_fn(self.agent_state, agent_feed)
        action = agent_out.planner_out.action
        self.env_feed.action = action
        return StepSample(
            frame=env_out.frame,
            last_reward=env_out.reward,
            is_first=env_out.is_first,
            is_last=env_out.is_last,
            to_play=env_out.to_play,
            legal_actions_mask=env_out.legal_actions,
            root_value=agent_out.planner_out.root_value,
            action_probs=agent_out.planner_out.action_probs,
            action=action,
        )


class TrainingWorker:
    def __init__(
        self,
        index: int,
        env_name: str,
        num_envs: int,
        model: NNModel,
        agent: Agent,
        planner: Planner,
        num_steps: int,
        seed: Optional[int] = 0,
        save_gif: bool = True,
    ):
        if seed is None:
            seed = index
        self.index = index
        self.seed = seed
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.save_gif = save_gif

        random_key = jax.random.PRNGKey(self.seed)
        model_key, agent_key = jax.random.split(random_key, 2)
        params, state = model.init_params_and_state(model_key)
        self.aei = AgentEnvironmentInterface(
            env_name=env_name,
            agent=agent,
            planner=planner,
            params=params,
            state=state,
            random_key=agent_key,
            num_envs=num_envs,
        )
        print("shape", self.aei.env.envs[0].spec.observations.observation.shape)
        self.vis = BreakthroughVisualizer(5, 6)
        self.traj_collector = TrajectoryCollector(num_envs)

    def run(self) -> List[TrajectorySample]:
        step_samples = []
        for _ in range(self.num_steps):
            step_sample = self.aei.tick()
            step_samples.append(step_sample)


class TrajectoryCollector:
    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.buffer: List[List[StepSample]] = [[] for _ in range(batch_size)]
        self.trajs: List[TrajectorySample] = []

    def add_step_sample(self, step_sample: StepSample) -> None:
        step_sample_flat = unstack_sequence_fields(step_sample, self.batch_size)
        for new_sample, step_samples in zip(step_sample_flat, self.buffer):
            step_samples.append(new_sample)
            if new_sample.is_last:
                traj = TrajectorySample.from_step_samples(step_samples)
                self.trajs.append(traj)
                step_samples.clear()

    def flush(self):
        ret = self.trajs.copy()
        self.trajs.clear()
        return ret


# %%
config = get_config()
model = get_model(config)
num_envs = 4
stacker = HistoryStacker(
    num_rows=config.env.num_rows,
    num_cols=config.env.num_cols,
    num_channels=config.env.num_channels,
    history_length=config.history_length,
    dim_action=config.dim_action,
)
agent = Agent(num_players=2, batch_size=num_envs, stacker=stacker)
planner = Planner(
    batch_size=num_envs,
    dim_action=config.dim_action,
    model=model,
    discount=config.discount,
    num_unroll_steps=config.num_unroll_steps,
    num_simulations=50,
    limit_depth=True,
)
# %%
tw = TrainingWorker(
    index=0,
    env_name=config.env.name,
    num_envs=4,
    model=model,
    agent=agent,
    planner=planner,
    num_steps=50,
)


# # %%
# for s in aei._step_samples:
#     display(aei.vis.make_image(s.frame[0]))

# # %%
# env_feed.action = np.random.choice(np.argwhere(env_out.legal_actions).ravel())

# # %%
# def sample_actions(legal_actions):
#     def choice(x):
#         return np.random.choice(np.argwhere(x).ravel())

#     return np.vectorize(choice, signature="(m)->()")(legal_actions)


# # %%
# env = GIIVecEnv(config.env.name, num_envs=num_envs)
# env_feed = env.init()
# for i in range(100):
#     env_out = env.step(env_feed)

#     env_feed.reset = env_out.is_last
#     # env_feed.action = np.random.choice(np.argwhere(env_out.legal_actions).ravel())
#     print(set(env_out.to_play))
#     # img = vis.make_image(env_out.frame[0, :])
#     # display(img)
