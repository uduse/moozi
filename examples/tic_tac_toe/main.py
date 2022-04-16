# %%
import numpy as np
import uuid
from functools import partial
from pathlib import Path

import moozi as mz
import ray
from absl import logging
from acme.jax.utils import weighted_softmax
from moozi.core import Config, Universe, link
from moozi.core.env import make_env
from moozi.laws import (
    OpenSpielEnvLaw,
    FrameStacker,
    TrajectoryOutputWriter,
    make_policy_feed,
)
from moozi.logging import JAXBoardLoggerActor, JAXBoardLoggerV2
from moozi.parameter_optimizer import ParameterOptimizer
from moozi.policy.mcts import ActionSampler, planner_law
from moozi.policy.mcts_core import (
    Node,
    SearchStrategy,
    anytree_display_in_notebook,
    anytree_filter_node,
    anytree_to_png,
    anytree_to_text,
    convert_to_anytree,
)
from moozi.replay import ReplayBuffer
from moozi.rollout_worker import RolloutWorkerWithWeights
from moozi.utils import WallTimer, check_ray_gpu, is_notebook
from tqdm import tqdm

from utils import *

ray.init(ignore_reinit_error=True, num_cpus=10, num_gpus=2)

# %%
num_epochs = 50

config = Config()
config.env = f"tic_tac_toe"
config.batch_size = 128
config.discount = 0.99
config.num_unroll_steps = 2
config.num_td_steps = 100
config.num_stacked_frames = 1
config.lr = 3e-3
config.replay_max_size = 100000
config.num_epochs = num_epochs
config.num_ticks_per_epoch = 12
config.num_samples_per_update = 30
config.num_env_workers = 5
config.num_universes_per_env_worker = 30
config.weight_decay = 5e-2
config.nn_arch_cls = mz.nn.ResNetArchitecture

# %%
env_spec = mz.make_env_spec(config.env)
single_frame_shape = env_spec.observations.observation.shape
obs_channels = single_frame_shape[-1] * config.num_stacked_frames
repr_channels = 4
dim_action = env_spec.actions.num_values

# %%
obs_rows, obs_cols, obs_channels = env_spec.observations.observation.shape

# %%
config.nn_spec = mz.nn.ResNetSpec(
    obs_rows=obs_rows,
    obs_cols=obs_cols,
    obs_channels=obs_channels,
    repr_rows=obs_rows,
    repr_cols=obs_cols,
    repr_channels=repr_channels,
    dim_action=dim_action,
    repr_tower_blocks=6,
    repr_tower_dim=4,
    pred_tower_blocks=6,
    pred_tower_dim=4,
    dyna_tower_blocks=6,
    dyna_tower_dim=4,
    dyna_state_blocks=6,
)
config.print()

num_interactions = (
    config.num_epochs
    * config.num_ticks_per_epoch
    * config.num_env_workers
    * config.num_universes_per_env_worker
)
print(f"num_interactions: {num_interactions}")

#  %%
def make_parameter_optimizer(config):
    param_opt = ray.remote(num_cpus=1, num_gpus=1)(ParameterOptimizer).remote()
    param_opt.make_training_suite.remote(config)
    param_opt.make_loggers.remote(
        lambda: [
            mz.logging.JAXBoardLoggerV2(name="param_opt", time_delta=15),
        ]
    ),
    return param_opt


# %%
def make_laws_train(config: Config):
    return [
        OpenSpielEnvLaw(make_env(config.env), num_players=2),
        FrameStacker(num_frames=config.num_stacked_frames, player=0),
        make_policy_feed,
        planner_law,
        ActionSampler(),
        TrajectoryOutputWriter(),
    ]


def make_workers_train(config: Config, param_opt: ParameterOptimizer):
    workers = []
    for _ in range(config.num_env_workers):
        worker = ray.remote(RolloutWorkerWithWeights).remote()
        worker.set_model.remote(param_opt.get_model.remote())
        worker.set_params_and_state.remote(param_opt.get_params_and_state.remote())
        worker.make_batching_layers.remote(config)
        worker.make_universes_from_laws.remote(
            partial(make_laws_train, config), config.num_universes_per_env_worker
        )
        workers.append(worker)
    return workers


def obs_to_ascii(obs):
    tokens = []
    for row in obs.reshape(3, 9).T:
        tokens.append(int(np.argwhere(row == 1)))
    tokens = np.array(tokens).reshape(3, 3)
    tokens = [row.tolist() for row in tokens]

    s = ""
    for row in tokens:
        for ele in row:
            if ele == 0:
                s += "."
            elif ele == 1:
                s += "O"
            else:
                s += "X"
        s += "\n"
    return s


def action_probs_to_ascii(action_probs):
    s = ""
    for row in action_probs.reshape(3, 3):
        for ele in row:
            s += f"{ele:.2f} "
        s += "\n"
    return s


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


def make_laws_eval(config: Config):
    return [
        OpenSpielEnvLaw(make_env(config.env), num_players=2),
        FrameStacker(num_frames=config.num_stacked_frames),
        make_policy_feed,
        planner_law,
        ActionSampler(temperature=0.3),
        log_evaluation_law,
    ]


def make_worker_eval(config: Config, param_opt: ParameterOptimizer):
    eval_worker = ray.remote(RolloutWorkerWithWeights).remote()
    eval_worker.set_model.remote(param_opt.get_model.remote())
    eval_worker.set_params_and_state.remote(param_opt.get_params_and_state.remote())
    eval_worker.make_universes_from_laws.remote(partial(make_laws_eval, config), 1)
    return eval_worker


# %%
def train(config: Config):
    param_opt = make_parameter_optimizer(config)
    replay_buffer = ray.remote(ReplayBuffer).remote(config)
    workers_train = make_workers_train(config, param_opt)
    jaxboard_logger = JAXBoardLoggerActor.remote("jaxboard_logger")
    Path(config.save_dir).mkdir(parents=True, exist_ok=True)

    with WallTimer():
        for epoch in range(config.num_epochs):
            logging.info(f"Epochs: {epoch + 1} / {config.num_epochs}")
            samples = [w.run.remote(config.num_ticks_per_epoch) for w in workers_train]
            samples_added = [replay_buffer.add_samples.remote(s) for s in samples]
            while samples_added:
                _, samples_added = ray.wait(samples_added)
                for _ in range(config.num_samples_per_update):
                    batch = replay_buffer.get_batch.remote(config.batch_size)
                    param_opt.update.remote(batch)

                for w in workers_train:
                    w.set_params_and_state.remote(
                        param_opt.get_params_and_state.remote()
                    )
            param_opt.save.remote(f"{config.save_dir}/param_opt_{epoch}.pkl")

            done = param_opt.log.remote()
            jaxboard_logger.write.remote(replay_buffer.get_stats.remote())

    ray.get(done)


# %%
def eval_param_pkl(param_pkl_path):
    logging.info(f"\n\nEvaluating {param_pkl_path}")
    if not param_opt:
        param_opt = make_parameter_optimizer(config)
    param_opt.restore.remote(Path(param_pkl_path))
    eval_worker = make_worker_eval(config, param_opt)
    for _ in range(50):
        ray.get(eval_worker.run.remote(1))
    logging.info("\n\n")


# %%
mode = None

if mode == "train":
    train(config)
elif mode == "eval":
    counter = 0
    while True:
        file_path = Path(config.save_dir) / Path(f"param_opt_{counter}.pkl")
        if file_path.exists():
            eval_param_pkl(file_path)
            counter += 1
        else:
            break
elif mode == 'eval_pool':
    pass

# elif mode == "train_eval":
#     train()
#     param_opt = make_parameter_optimizer()
#     counter = 0
#     while True:
#         file_path = Path(f"params_{counter}.pkl")
#         if file_path.exists():
#             eval_param_pkl(file_path, param_opt)
#             counter += 1
#         else:
#             break

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
