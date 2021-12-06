# %%
from functools import partial
from pathlib import Path

import moozi as mz
import ray
from absl import logging
from moozi.core import link
from moozi.core.link import Universe
from moozi.logging import JAXBoardStepData, MetricsReporterActor
from moozi.parameter_optimizer import ParameterOptimizer
from moozi.policy.mcts_core import (
    Node,
    SearchStrategy,
    anytree_display_in_notebook,
    anytree_filter_node,
    anytree_to_png,
    anytree_to_text,
    convert_to_anytree,
    get_next_player,
)
from moozi.replay import ReplayBuffer
from moozi.rollout_worker import RolloutWorkerWithWeights
from moozi.utils import WallTimer, is_notebook
from tqdm import tqdm

from utils import *

# %%
ray.init(ignore_reinit_error=True)


# %%
num_epochs = 100

config = mz.Config().update(
    env=f"tic_tac_toe",
    batch_size=256,
    discount=0.99,
    num_unroll_steps=5,
    num_td_steps=100,
    num_stacked_frames=2,
    lr=2e-3,
    replay_buffer_size=1_000_000,
    dim_repr=128,
    num_epochs=num_epochs,
    num_ticks_per_epoch=50,
    num_updates_per_samples_added=10,
    num_rollout_workers=8,
    num_rollout_universes_per_worker=50,
)

num_interactions = (
    config.num_epochs
    * config.num_ticks_per_epoch
    * config.num_rollout_workers
    * config.num_rollout_universes_per_worker
)
print(f"num_interactions: {num_interactions}")
config.print()


def make_rollout_worker_universes(
    self: RolloutWorkerWithWeights, config: Config
) -> List[UniverseAsync]:
    dim_actions = make_env_spec(config.env).actions.num_values

    def _make_universe(index):
        tape = Tape(index)
        planner_law = make_async_planner_law(
            init_inf_fn=lambda features: self.init_inf_fn_unbatched(
                self.params, features
            ),
            recurr_inf_fn=lambda features: self.recurr_inf_fn_unbatched(
                self.params, features
            ),
            dim_actions=dim_actions,
        )
        laws = [
            EnvironmentLaw(make_env(config.env), num_players=2),
            FrameStacker(num_frames=config.num_stacked_frames),
            set_policy_feed,
            planner_law,
            TrajectoryOutputWriter(),
        ]
        return UniverseAsync(tape, laws)

    universes = [
        _make_universe(i) for i in range(config.num_rollout_universes_per_worker)
    ]
    return universes


@link
def print_evaluation_law(obs, action_probs, action, to_play):
    print(obs_to_ascii(obs[0]))
    print(action_probs_to_ascii(action_probs))
    if to_play == 0:
        to_play_repr = "X"
    else:
        to_play_repr = "O"
    print(f"{to_play_repr} -> {action}")
    print("\n")


def make_evaluator_universes(
    self: RolloutWorkerWithWeights, config: Config
) -> List[UniverseAsync]:
    dim_actions = make_env_spec(config.env).actions.num_values

    def _make_universe():
        tape = Tape(0)
        planner_law = make_async_planner_law(
            init_inf_fn=lambda features: self.init_inf_fn_unbatched(
                self.params, features
            ),
            recurr_inf_fn=lambda features: self.recurr_inf_fn_unbatched(
                self.params, features
            ),
            dim_actions=dim_actions,
            num_simulations=30,
            include_tree=True,
        )
        laws = [
            EnvironmentLaw(make_env(config.env), num_players=2),
            FrameStacker(num_frames=config.num_stacked_frames),
            set_policy_feed,
            planner_law,
            print_evaluation_law,
            link(lambda mcts_root: dict(output_buffer=(mcts_root,))),
        ]
        return UniverseAsync(tape, laws)

    return [_make_universe()]


# %%
load_if_possible = False
path = Path("dump.pkl")
if load_if_possible and path.exists():
    param_opt = ray.remote(ParameterOptimizer).restore.remote(path)
    print("restored from", path)
else:
    param_opt = ray.remote(ParameterOptimizer).remote()
    param_opt.build.remote(partial(make_param_opt_properties, config=config)),
    param_opt.build_loggers.remote(
        lambda: [
            # "print",
            # TerminalLogger(label="Parameter Optimizer", print_fn=print),
            mz.logging.JAXBoardLogger(name="param_opt"),
        ]
    ),
    param_opt.log_stats.remote()

# %%
replay_buffer = ray.remote(ReplayBuffer).remote(config)

# %%
def build_worker(config, param_opt):
    worker = ray.remote(RolloutWorkerWithWeights).remote()
    worker.set_network.remote(param_opt.get_network.remote())
    worker.set_params.remote(param_opt.get_params.remote())
    worker.build_batching_layers.remote(
        partial(make_rollout_worker_batching_layers, config=config)
    )
    worker.build_universes.remote(partial(make_rollout_worker_universes, config=config))
    return worker


rollout_workers = [
    build_worker(config, param_opt) for _ in range(config.num_rollout_workers)
]

# evaluator = ray.remote(RolloutWorkerWithWeights).remote()
# evaluator.set_network.remote(param_opt.get_network.remote())
# evaluator.set_params.remote(param_opt.get_params.remote())
# evaluator.build_universes.remote(partial(make_evaluator_universes, config=config))

# %%
with WallTimer():
    for epoch in range(config.num_epochs):
        logging.info(f"Epochs: {epoch + 1} / {config.num_epochs}")

        samples = [w.run.remote(config.num_ticks_per_epoch) for w in rollout_workers]
        samples_added = [replay_buffer.add_samples.remote(s) for s in samples]
        # evaluation_done = evaluator.run.remote(50)
        while samples_added:
            _, samples_added = ray.wait(samples_added)
            for _ in range(config.num_updates_per_samples_added):
                batch = replay_buffer.get_batch.remote(config.batch_size)
                param_opt.update.remote(batch)

            for w in rollout_workers:
                w.set_params.remote(param_opt.get_params.remote())

            param_opt.log_stats.remote()
            path = Path(f"params_{epoch}.pkl")
            param_opt.save.remote(path)

    # ray.get(evaluation_done)
