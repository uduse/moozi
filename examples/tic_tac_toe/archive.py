# %%
from dataclasses import dataclass
from pathlib import Path
from functools import partial

import moozi as mz

# import ray
from absl import logging
from moozi.core import link
from moozi.core.link import Universe

# from moozi.laws import PlayerFrameStacker
from moozi.logging import JAXBoardLogger, JAXBoardStepData, MetricsReporterActor
from moozi.parameter_optimizer import ParameterOptimizer
from moozi.replay import ReplayBuffer
from moozi.utils import WallTimer, is_notebook
from moozi.rollout_worker import RolloutWorkerWithWeights

from utils import *

from tqdm import tqdm

# ray.init(ignore_reinit_error=True)

from moozi.policy.mcts_core import (
    Node,
    anytree_to_png,
    anytree_to_text,
    convert_to_anytree,
    anytree_display_in_notebook,
    anytree_filter_node,
    get_next_player,
    SearchStrategy,
)

# %%
config = mz.Config().update(
    env=f"tic_tac_toe",
    batch_size=256,
    discount=0.99,
    num_unroll_steps=3,
    num_td_steps=100,
    num_stacked_frames=2,
    lr=2e-3,
    replay_buffer_size=1_000_000,
    dim_repr=128,
    num_epochs=15000,
    num_ticks_per_epoch=100,
    num_updates_per_samples_added=10,
    num_rollout_workers=1,
    num_rollout_universes_per_worker=1,
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
            root_inf_fn=lambda features: self.root_inf_1(
                self.params, features
            ),
            trans_inf_fn=lambda features: self.trans_inf_unbatched(
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


def evaluation_to_str(obs, action_probs, action, to_play):
    s = ""
    s += obs_to_ascii(obs[0]) + "\n"
    s += action_probs_to_ascii(action_probs) + "\n"
    if to_play == 0:
        to_play_repr = "X"
    else:
        to_play_repr = "O"
    s += f"{to_play_repr} -> {action}" + "\n\n"
    return s


@link
def print_evaluation_law(obs, action_probs, action, to_play):
    print(evaluation_to_str(obs, action_probs, action, to_play))


@link
@dataclass
class EvaluationWrite:
    logger: JAXBoardLogger = JAXBoardLogger("evaluation", log_dir=Path("evaluation"))

    def write(self, obs, action_probs, action, to_play):
        evalution_str = evaluation_to_str(obs, action_probs, action, to_play)
        self.logger.write(evalution_str)


def make_evaluator_universes(
    self: RolloutWorkerWithWeights, config: Config
) -> List[UniverseAsync]:
    dim_actions = make_env_spec(config.env).actions.num_values

    def _make_universe():
        tape = Tape(0)
        planner_law = make_async_planner_law(
            root_inf_fn=lambda features: self.root_inf_1(
                self.params, features
            ),
            trans_inf_fn=lambda features: self.trans_inf_unbatched(
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
    param_opt = ParameterOptimizer.restore(path)
    print("restored from", path)
else:
    param_opt = ParameterOptimizer()
    param_opt.build(partial(make_param_opt_properties, config=config)),
    param_opt.build_loggers(
        lambda: [
            # "print",
            # TerminalLogger(label="Parameter Optimizer", print_fn=print),
            mz.logging.JAXBoardLogger(name="param_opt"),
        ]
    ),
    param_opt.log_stats()

# %%
replay_buffer = ReplayBuffer(config)

# %%
worker = RolloutWorkerWithWeights()
worker.set_model(param_opt.get_network())
worker.set_params(param_opt.get_params())
worker.build_universes(partial(make_rollout_worker_universes, config=config))

# %%
for i in tqdm(range(config.num_epochs)):
    # print(f"Epoch {i}")
    samples = worker.run(config.num_ticks_per_epoch)
    replay_buffer.add_samples(samples)
    batch = replay_buffer.get_batch(config.batch_size)
    param_opt.update(batch)
    worker.set_params(param_opt.get_params())

# %%
evaluator = RolloutWorkerWithWeights()
evaluator.set_model(param_opt.get_network())
evaluator.set_params(param_opt.get_params())
evaluator.build_universes(partial(make_evaluator_universes, config=config))

for _ in range(50):
    mcts_root = evaluator.run(1)[0]
    mcts_root_anytree = convert_to_anytree(mcts_root)
    anytree_filter_node(mcts_root_anytree, lambda n: n.visits > 0)
    anytree_display_in_notebook(mcts_root_anytree)

# %%
for logger in param_opt.loggers:
    if hasattr(logger, "close"):
        logger.close()


    