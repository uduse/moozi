# %%
import uuid
from functools import partial
from pathlib import Path

from acme.jax.utils import weighted_softmax

import moozi as mz
import ray
from absl import logging
from moozi.core import link
from moozi.core.link import Universe
from moozi.logging import (
    JAXBoardLoggerActor,
    JAXBoardLoggerV2,
    JAXBoardStepData,
)
from moozi.parameter_optimizer import ParameterOptimizer
from moozi.policy.mcts_async import ActionSamplerLaw, make_async_planner_law
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
from moozi.utils import WallTimer, check_ray_gpu, is_notebook
from tqdm import tqdm

from utils import *

ray.init(ignore_reinit_error=True, num_cpus=10, num_gpus=2)

# %%
num_epochs = 300
config = mz.Config().update(
    env=f"tic_tac_toe",
    batch_size=256,
    discount=1.0,
    num_unroll_steps=3,
    num_td_steps=100,
    num_stacked_frames=1,
    weight_decay=3e-4,
    lr=3e-3,
    replay_buffer_size=1_000_000,
    dim_repr=64,
    num_epochs=num_epochs,
    num_ticks_per_epoch=30,
    num_updates_per_samples_added=10,
    num_rollout_workers=10,
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


def make_train_worker_universes(
    self: RolloutWorkerWithWeights, config: Config
) -> List[UniverseAsync]:
    dim_actions = make_env_spec(config.env).actions.num_values

    def _make_universe(index):
        tape = Tape(index)
        planner_law = make_async_planner_law(
            root_inf_fn=lambda features: self.root_inf_unbatched(
                self.params, features
            ),
            trans_inf_fn=lambda features: self.trans_inf_unbatched(
                self.params, features
            ),
            dim_actions=dim_actions,
            num_simulations=10,
        )
        laws = [
            EnvironmentLaw(make_env(config.env), num_players=2),
            FrameStacker(num_frames=config.num_stacked_frames),
            make_policy_feed,
            planner_law,
            ActionSamplerLaw(),
            TrajectoryOutputWriter(),
        ]
        return UniverseAsync(tape, laws)

    universes = [
        _make_universe(i) for i in range(config.num_rollout_universes_per_worker)
    ]
    return universes


def evaluation_to_str(obs, action_probs, action, to_play, mcts_root):
    s = ""
    s += obs_to_ascii(obs[0]) + "\n\n"
    s += action_probs_to_ascii(action_probs) + "\n\n"
    if to_play == 0:
        to_play_repr = "X"
    else:
        to_play_repr = "O"
    s += f"{to_play_repr} -> {action}" + "\n\n"

    anytree_root = convert_to_anytree(mcts_root)
    anytree_filter_node(anytree_root, lambda n: n.visits > 0)
    s += "012\n345\n678\n\n"
    s += anytree_to_text(anytree_root) + "\n\n\n"

    fname = str(uuid.uuid4())
    anytree_to_png(anytree_root, Path(f"/tmp/{fname}.png"))
    s += f"<img src='/tmp/{fname}.png'>"

    return s


@link
def log_evaluation_law(obs, action_probs, action, to_play, mcts_root):
    s = evaluation_to_str(obs, action_probs, action, to_play, mcts_root)
    logging.info("\n" + s)


def make_eval_worker_universes(
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
        )
        laws = [
            EnvironmentLaw(make_env(config.env), num_players=2),
            FrameStacker(num_frames=config.num_stacked_frames),
            make_policy_feed,
            planner_law,
            ActionSamplerLaw(temperature=0.5),
            log_evaluation_law,
        ]
        return UniverseAsync(tape, laws)

    return [_make_universe()]


def make_parameter_optimizer():
    param_opt = ray.remote(num_gpus=1)(ParameterOptimizer).remote()
    param_opt.build.remote(partial(make_param_opt_properties, config=config)),
    param_opt.build_loggers.remote(
        lambda: [
            # "print",
            # TerminalLogger(label="Parameter Optimizer", print_fn=print),
            mz.logging.JAXBoardLogger(name="param_opt", time_delta=15),
        ]
    ),
    param_opt.log_stats.remote()
    return param_opt


def make_replay_buffer():
    replay_buffer = ray.remote(ReplayBuffer).remote(config)
    return replay_buffer


def build_worker_train(config, param_opt):
    worker = ray.remote(RolloutWorkerWithWeights).remote()
    worker.set_network.remote(param_opt.get_network.remote())
    worker.set_params.remote(param_opt.get_params.remote())
    worker.build_batching_layers.remote(
        partial(make_rollout_worker_batching_layers, config=config)
    )
    worker.build_universes.remote(partial(make_train_worker_universes, config=config))
    return worker


def make_train_rollout_workers(param_opt):
    return [
        build_worker_train(config, param_opt) for _ in range(config.num_rollout_workers)
    ]


def make_eval_rollout_workers(param_opt):
    eval_worker = ray.remote(RolloutWorkerWithWeights).remote()
    eval_worker.set_network.remote(param_opt.get_network.remote())
    eval_worker.set_params.remote(param_opt.get_params.remote())
    eval_worker.build_universes.remote(
        partial(make_eval_worker_universes, config=config)
    )
    return eval_worker


# %%
def train():
    replay_buffer = make_replay_buffer()
    param_opt = make_parameter_optimizer()
    train_workers = make_train_rollout_workers(param_opt)
    reporter = ray.remote(JAXBoardLoggerV2).remote(name="reporter")

    with WallTimer():
        for epoch in tqdm(range(config.num_epochs)):
            logging.info(f"Epochs: {epoch + 1} / {config.num_epochs}")

            samples = [w.run.remote(config.num_ticks_per_epoch) for w in train_workers]
            samples_added = [replay_buffer.add_samples.remote(s) for s in samples]
            while samples_added:
                _, samples_added = ray.wait(samples_added)
                for _ in range(config.num_updates_per_samples_added):
                    batch = replay_buffer.get_batch.remote(config.batch_size)
                    param_opt.update.remote(batch)

                for w in train_workers:
                    w.set_params.remote(param_opt.get_params.remote())

                param_opt.log_stats.remote()
                path = Path(f"params_{epoch}.pkl")
                reporter.write.remote(replay_buffer.get_logger_data.remote())
                param_opt.save.remote(path)


# %%
def eval(param_pkl_path, param_opt=None):
    logging.info(f"\n\nEvaluating {param_pkl_path}")
    if not param_opt:
        param_opt = make_parameter_optimizer()
    param_opt.restore.remote(Path(param_pkl_path))
    eval_worker = make_eval_rollout_workers(param_opt)
    for _ in range(50):
        ray.get(eval_worker.run.remote(1))
    logging.info("\n\n")


# %%
mode = None
eval_path = None

if mode == "train":
    train()
elif mode == "eval":
    assert eval_path is not None
    eval_path = Path(eval_path)
    assert eval_path.exists()
    param_opt = make_parameter_optimizer()
    if eval_path.is_dir():
        counter = 0
        while True:
            file_path = eval_path / Path(f"params_{counter}.pkl")
            if file_path.exists():
                eval(file_path, param_opt)
                counter += 1
            else:
                break
    else:
        eval(eval_path, param_opt)
elif mode == "train_eval":
    train()
    param_opt = make_parameter_optimizer()
    counter = 0
    while True:
        file_path = Path(f"params_{counter}.pkl")
        if file_path.exists():
            eval(file_path, param_opt)
            counter += 1
        else:
            break

# %%
### Evaluation
# param_opt = ray.remote(ParameterOptimizer).remote()
# param_opt.build.remote(partial(make_param_opt_properties, config=config)),
# param_opt.restore.remote(Path("/root/.local/share/virtualenvs/moozi-g1CZ00E9/.guild/runs/75e23c33b9dc4d938fd1445da5938d92/params_48.pkl"))
# param_opt.log_stats.remote()

# # %%
# evaluator = RolloutWorkerWithWeights()
# evaluator.set_network(ray.get(param_opt.get_network.remote()))
# evaluator.set_params(ray.get(param_opt.get_params.remote()))
# evaluator.build_universes(partial(make_evaluator_universes, config=config))

# # %%
# node = evaluator.run(1)[0]
# anytree_node = convert_to_anytree(node)
# anytree_filter_node(anytree_node, lambda n: n.visits > 0)
# print('012\n345\n678')
# anytree_display_in_notebook(anytree_node)
# # %%