# %%
from pathlib import Path
from PIL import Image, ImageOps

# from tqdm.notebook import tqdm
from tqdm import tqdm
import moozi
from functools import partial
import pyspiel
from dataclasses import dataclass, field
import tree
from IPython.display import display
import jax
import jax.numpy as jnp
import numpy as np
import chex
from flax import struct
import random
from moozi.core import make_env, StepSample, TrajectorySample
from moozi.core.types import BASE_PLAYER
from moozi.core.utils import (
    stack_sequence_fields,
    unstack_sequence_fields,
    HistoryStacker,
    unstack_sequence_fields_pytree,
)
from moozi.core.vis import BreakthroughVisualizer, save_gif
from moozi.core.env import GIIEnv, GIIVecEnv, GIIEnvFeed, GIIEnvOut
from moozi.nn import RootFeatures, NNModel
from moozi.nn.training import make_target_from_traj
from moozi.planner import Planner
from moozi.replay import ReplayBuffer
from moozi.parameter_optimizer import ParameterServer
from lib import get_model, get_config

# %%
from typing import Callable, Dict, Tuple, Optional, List, Union, Any
import haiku as hk


class PolicyFeed(struct.PyTreeNode):
    stacker_state: HistoryStacker.StackerState
    params: hk.Params
    state: hk.State
    env_out: GIIEnvOut
    last_action: chex.Array
    random_key: chex.PRNGKey

    stacker: HistoryStacker = struct.field(pytree_node=False)
    planner: Planner = struct.field(pytree_node=False)


class PolicyOut(struct.PyTreeNode):
    stacker_state: HistoryStacker.StackerState
    planner_out: Planner.PlannerOut


PolicyType = Callable[[PolicyFeed], PolicyOut]


def policy(
    policy_feed: PolicyFeed,
) -> PolicyOut:
    stacker_state = jax.vmap(policy_feed.stacker.apply)(
        state=policy_feed.stacker_state,
        frame=policy_feed.env_out.frame,
        action=policy_feed.last_action,
        is_first=policy_feed.env_out.is_first,
    )
    root_feats = RootFeatures(
        stacker_state.frames,
        stacker_state.actions,
        policy_feed.env_out.to_play,
    )
    planner_feed = Planner.PlannerFeed(
        params=policy_feed.params,
        state=policy_feed.state,
        root_feats=root_feats,
        legal_actions=policy_feed.env_out.legal_actions,
        random_key=policy_feed.random_key,
    )
    planner_out = policy_feed.planner.run(planner_feed)
    return PolicyOut(stacker_state=stacker_state, planner_out=planner_out)


class AgentEnvironmentInterface:
    def __init__(
        self,
        env_name: str,
        stacker: HistoryStacker,
        planner: Union[Planner, Dict[int, Planner]],
        params: Union[hk.Params, Dict[int, hk.Params]],
        state: Union[hk.State, Dict[int, hk.State]],
        random_key: chex.PRNGKey,
        num_envs: int = 1,
    ):
        self.env = GIIVecEnv(env_name, num_envs=num_envs)
        self.env_feed = self.env.init()
        self.env_out: Optional[GIIEnvOut] = None

        self.stacker = stacker
        self.stacker_state = stacker.init()

        self.random_key = random_key
        self.planner = planner
        self.params = params
        self.state = state

        self.policy: PolicyType = jax.jit(policy, backend="gpu")

    @staticmethod
    def _select_for_player(data: Union[Any, Dict[int, Any]], to_play: int):
        if to_play == pyspiel.PlayerId.TERMINAL:
            to_play = 0
        if isinstance(data, dict) and (to_play in data):
            return data[to_play]
        else:
            return data

    def _select_planner(self, to_play: int) -> Planner:
        return self._select_for_player(self.planner, to_play)

    def _select_params_and_state(self, to_play: int) -> Tuple[hk.Params, hk.State]:
        params = self._select_for_player(self.params, to_play)
        state = self._select_for_player(self.state, to_play)
        return params, state

    def _next_key(self):
        self.random_key, next_key = jax.random.split(self.random_key)
        return next_key

    def tick(self) -> StepSample:
        env_out = self.env.step(self.env_feed)
        to_play = int(env_out.to_play)

        params, state = self._select_params_and_state(to_play)
        planner = self._select_planner(to_play)

        policy_feed = PolicyFeed(
            params=params,
            state=state,
            planner=planner,
            env_out=env_out,
            last_action=self.env_feed.action,
            stacker=self.stacker,
            stacker_state=self.stacker_state,
            random_key=self._next_key(),
        )
        policy_out = self.policy(policy_feed)
        action = policy_out.planner_out.action

        self.stacker_state = policy_out.stacker_state
        self.env_feed.reset = env_out.is_last
        self.env_feed.action = np.array(action)
        self.env_out = env_out

        return StepSample(
            frame=env_out.frame,
            last_reward=env_out.reward,
            is_first=env_out.is_first,
            is_last=env_out.is_last,
            to_play=env_out.to_play,
            legal_actions_mask=env_out.legal_actions,
            root_value=policy_out.planner_out.root_value,
            action_probs=policy_out.planner_out.action_probs,
            action=action,
        )


class TrainingWorker:
    def __init__(
        self,
        index: int,
        env_name: str,
        num_envs: int,
        model: NNModel,
        stacker: HistoryStacker,
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
            stacker=stacker,
            planner=planner,
            params=params,
            state=state,
            random_key=agent_key,
            num_envs=num_envs,
        )
        self.vis = BreakthroughVisualizer(5, 6)
        self.traj_collector = TrajectoryCollector(num_envs)

    def run(self) -> List[TrajectorySample]:
        samples = [self.aei.tick() for _ in range(self.num_steps)]
        self.traj_collector.add_step_samples(samples)
        return self.traj_collector.flush()


class TrajectoryCollector:
    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.buffer: List[List[StepSample]] = [[] for _ in range(batch_size)]
        self.trajs: List[TrajectorySample] = []

    def add_step_sample(self, step_sample: StepSample) -> "TrajectoryCollector":
        step_sample_flat = unstack_sequence_fields_pytree(step_sample, self.batch_size)
        for new_sample, step_samples in zip(step_sample_flat, self.buffer):
            step_samples.append(new_sample)
            if new_sample.is_last:
                traj = TrajectorySample.from_step_samples(step_samples)
                self.trajs.append(traj)
                step_samples.clear()
        return self

    def add_step_samples(self, step_samples: List[StepSample]) -> "TrajectoryCollector":
        for s in step_samples:
            self.add_step_sample(s)
        return self

    def flush(self) -> List[TrajectorySample]:
        ret = self.trajs.copy()
        self.trajs.clear()
        return ret


# %%
def training_suite_factory(config):
    from moozi.core.scalar_transform import make_scalar_transform
    from moozi.nn.training import make_training_suite

    scalar_transform = make_scalar_transform(**config.scalar_transform)
    nn_arch_cls = eval(config.nn.arch_cls)
    nn_spec = eval(config.nn.spec_cls)(
        **config.nn.spec_kwargs,
        scalar_transform=scalar_transform,
    )
    return partial(
        make_training_suite,
        seed=config.seed,
        nn_arch_cls=nn_arch_cls,
        nn_spec=nn_spec,
        weight_decay=config.train.weight_decay,
        lr=config.train.lr,
        num_unroll_steps=config.num_unroll_steps,
        history_length=config.history_length,
        target_update_period=config.train.target_update_period,
        consistency_loss_coef=config.train.consistency_loss_coef,
    )


# %%
import mctx
from typing import Sequence
import pygraphviz


def convert_tree_to_graph(
    tree: mctx.Tree,
    action_labels: Optional[Sequence[str]] = None,
    batch_index: int = 0,
    show_only_expanded: bool = True,
    image_path: Optional[str] = None,
    image_suffix: str = "png",
) -> pygraphviz.AGraph:
    """Converts a search tree into a Graphviz graph.
    Args:
      tree: A `Tree` containing a batch of search data.
      action_labels: Optional labels for edges, defaults to the action index.
      batch_index: Index of the batch element to plot.
    Returns:
      A Graphviz graph representation of `tree`.

    Copy-pasted from mctx library examples.
    https://github.com/deepmind/mctx/blob/main/examples/visualization_demo.py
    """
    chex.assert_rank(tree.node_values, 2)
    batch_size = tree.node_values.shape[0]
    if action_labels is None:
        action_labels = list(map(str, range(tree.num_actions)))
    elif len(action_labels) != tree.num_actions:
        raise ValueError(
            f"action_labels {action_labels} has the wrong number of actions "
            f"({len(action_labels)}). "
            f"Expecting {tree.num_actions}."
        )

    def node_to_str(node_i, reward=0, discount=1):
        return (
            f"I: {node_i}\n"
            f"R: {reward:.2f}\n"
            # f"d: {discount:.2f}\n"
            f"V: {tree.node_values[batch_index, node_i]:.2f}\n"
            f"N: {tree.node_visits[batch_index, node_i]}\n"
        )

    def edge_to_str(node_i, a_i):
        node_index = jnp.full([batch_size], node_i)
        probs = jax.nn.softmax(tree.children_prior_logits[batch_index, node_i])
        return (
            f"A: {action_labels[a_i]}\n"
            f"Q: {tree.qvalues(node_index)[batch_index, a_i]:.2f}\n"
            f"p: {probs[a_i]:.2f}\n"
        )

    graph = pygraphviz.AGraph(directed=True, imagepath=image_path)
    graph.node_attr.update(
        shape="box", height=2.0, imagepos="tc", labelloc="b", fontname="Courier New"
    )

    # Add root
    graph.add_node(
        0,
        label=node_to_str(node_i=0),
        **({"image": f"0.{image_suffix}"} if image_path else {}),
    )
    # Add all other nodes and connect them up.
    for node_id in range(tree.num_simulations):
        for action_id in range(tree.num_actions):
            # Index of children, or -1 if not expanded
            child_id = tree.children_index[batch_index, node_id, action_id]
            if show_only_expanded:
                to_show = child_id >= 0
            else:
                to_show = True
            if to_show:
                graph.add_node(
                    child_id,
                    label=node_to_str(
                        node_i=child_id,
                        reward=tree.children_rewards[batch_index, node_id, action_id],
                        discount=tree.children_discounts[
                            batch_index, node_id, action_id
                        ],
                    ),
                    **{"image": f"{child_id}.{image_suffix}"} if image_path else {},
                )
                graph.add_edge(node_id, child_id, label=edge_to_str(node_id, action_id))

    return graph


def save_state_to_image(vis, game_state, i):
    if game_state:
        target_shape = game_state.get_game().observation_tensor_shape()
        frame = np.array(game_state.observation_tensor(BASE_PLAYER))
        frame = np.moveaxis(frame.reshape(target_shape), 0, -1)
        img = vis.make_image(frame)
        img = ImageOps.contain(img, (120, 120), Image.Resampling.LANCZOS)
    else:
        img = Image.new("RGBA", (120, 120), color="black")
    with open(f"/home/zeyi/assets/imgs/{i}.png", "wb") as f:
        img.save(f, format="PNG")


# %%
from moozi.nn import TransitionFeatures
from moozi.planner import qtransform_by_parent_and_siblings_inherit


def _view_from_player(scalar: chex.Array, to_play: chex.Array):
    return jax.lax.select(to_play == BASE_PLAYER, scalar, -scalar)


def _next_player(player: chex.Array):
    return jnp.logical_not(player).astype(player.dtype)


def make_paritial_recurr_fn(model, state, discount):
    def recurr_fn(params, random_key, action, embedding):
        hidden_state, to_play = embedding
        trans_feats = TransitionFeatures(hidden_state, action)
        is_training = False
        nn_output, _ = model.trans_inference(params, state, trans_feats, is_training)
        chex.assert_shape(nn_output.reward, (None,))
        chex.assert_shape(nn_output.value, (None,))
        rnn_output = mctx.RecurrentFnOutput(
            reward=_view_from_player(nn_output.reward, to_play),
            discount=jnp.full_like(nn_output.reward, fill_value=discount),
            prior_logits=nn_output.policy_logits,
            value=_view_from_player(nn_output.value, to_play),
        )
        return rnn_output, (nn_output.hidden_state, _next_player(to_play))

    return recurr_fn


class Planner(struct.PyTreeNode):
    batch_size: int
    dim_action: int
    model: NNModel = struct.field(pytree_node=False)
    discount: float = 1.0
    num_unroll_steps: int = struct.field(pytree_node=False, default=5)
    num_simulations: int = struct.field(pytree_node=False, default=10)
    limit_depth: bool = struct.field(pytree_node=False, default=True)
    use_gumbel: bool = struct.field(pytree_node=False, default=True)

    class PlannerFeed(struct.PyTreeNode):
        params: hk.Params
        state: hk.State
        root_feats: RootFeatures
        legal_actions: chex.Array
        random_key: chex.PRNGKey

    class PlannerOut(struct.PyTreeNode):
        action: Optional[chex.ArrayDevice]
        action_probs: chex.Array
        tree: Optional[mctx.Tree]
        prior_probs: chex.Array
        visit_counts: chex.Array
        q_values: chex.Array
        root_value: chex.Array

    def run(self, feed: "PlannerFeed") -> "PlannerOut":
        is_training = False
        nn_output, _ = self.model.root_inference(
            feed.params, feed.state, feed.root_feats, is_training
        )
        root = mctx.RootFnOutput(
            prior_logits=nn_output.policy_logits,
            value=_view_from_player(nn_output.value, feed.root_feats.to_play),
            embedding=(nn_output.hidden_state, feed.root_feats.to_play),
        )
        invalid_actions = jnp.logical_not(feed.legal_actions)

        if self.use_gumbel:
            mctx_out = mctx.gumbel_muzero_policy(
                params=feed.params,
                rng_key=feed.random_key,
                root=root,
                recurrent_fn=make_paritial_recurr_fn(
                    self.model, feed.state, self.discount
                ),
                num_simulations=self.num_simulations,
                max_depth=self.num_unroll_steps if self.limit_depth else None,
                invalid_actions=invalid_actions,
                max_num_considered_actions=16,
            )
        else:
            mctx_out = mctx.muzero_policy(
                params=feed.params,
                rng_key=feed.random_key,
                root=root,
                recurrent_fn=make_paritial_recurr_fn(
                    self.model, feed.state, self.discount
                ),
                num_simulations=self.num_simulations,
                max_depth=self.num_unroll_steps if self.limit_depth else None,
                invalid_actions=invalid_actions,
                qtransform=qtransform_by_parent_and_siblings_inherit,
            )

        action = mctx_out.action
        stats = mctx_out.search_tree.summary()
        prior_probs = jax.nn.softmax(nn_output.policy_logits)
        visit_counts = stats.visit_counts
        action_probs = mctx_out.action_weights
        q_values = stats.qvalues
        root_value = stats.value
        # if self.output_tree:
        tree = mctx_out.search_tree
        # else:
        #     tree = None

        return self.PlannerOut(
            action=action,
            action_probs=action_probs,
            tree=tree,
            prior_probs=prior_probs,
            visit_counts=visit_counts,
            q_values=q_values,
            root_value=root_value,
        )


# %%
config = get_config()
model = get_model(config)
num_envs = 16
stacker = HistoryStacker(
    num_rows=config.env.num_rows,
    num_cols=config.env.num_cols,
    num_channels=config.env.num_channels,
    history_length=config.history_length,
    dim_action=config.dim_action,
)
planner = Planner(
    batch_size=num_envs,
    dim_action=config.dim_action,
    model=model,
    discount=config.discount,
    num_unroll_steps=config.num_unroll_steps,
    num_simulations=1,
    limit_depth=True,
)

# %%
rb = ReplayBuffer(**config.replay.kwargs)
ps = ParameterServer(training_suite_factory(config))
tw = TrainingWorker(
    index=0,
    env_name=config.env.name,
    num_envs=num_envs,
    model=model,
    agent=agent,
    planner=planner,
    num_steps=50,
)

# %%
for i in range(500):
    trajs = tw.run()
    rb.add_trajs(trajs)
    batch = rb.get_train_targets_batch(1024)
    loss = ps.update(batch, 256)
    ps.log_tensorboard(i)
    if i % 10 == 0:
        ps.save()
    print(f"{loss=}")

# %%
rb.get_stats()

# %%
params = ps.training_state.params
state = ps.training_state.state

# %%
aei = AgentEnvironmentInterface(
    env_name=config.env.name,
    stacker=HistoryStacker(
        num_rows=config.env.num_rows,
        num_cols=config.env.num_cols,
        num_channels=config.env.num_channels,
        history_length=config.history_length,
        dim_action=config.dim_action,
    ),
    planner=Planner(
        batch_size=1,
        dim_action=config.dim_action,
        model=model,
        discount=-1,
        num_unroll_steps=5,
        num_simulations=100,
        limit_depth=True,
    ),
    params=params,
    state=state,
    random_key=jax.random.PRNGKey(0),
)

# %%
for i in range(10):
    aei.tick()

# %%
aei.tick()
tw.vis.make_image(aei.env_out.frame[0])

# %%
root_state = aei.env.envs[0]._backend.get_state
search_tree = aei.agent_out.planner_out.tree
search_tree = jax.tree_util.tree_map(
    lambda x: jax.device_put(x, device=jax.devices("cpu")[0]), search_tree
)
# %%
node_states = {0: root_state}
for node_id in tqdm(range(search_tree.num_simulations)):
    for action_id in range(search_tree.num_actions):
        child_id = int(search_tree.children_index[0, node_id, action_id])
        if child_id < 0:
            continue
        # print(node_id, child_id)
        if node_states[node_id] and (action_id != 0):
            new_state = node_states[node_id].clone()
            new_state.apply_action(action_id - 1)
            node_states[child_id] = new_state
        else:
            node_states[child_id] = None

# %%
image_path = Path("/home/zeyi/assets/imgs")
for key, game_state in node_states.items():
    save_state_to_image(tw.vis, game_state, key)

# %%
g = convert_tree_to_graph(search_tree, image_path=str(image_path))
g.write("/home/zeyi/assets/graph.dot")
