# %%
from loguru import logger
import ray
from functools import partial
from moozi import rollout_worker

from moozi.logging import JAXBoardLoggerRemote, TerminalLoggerRemote
from moozi.replay import ReplayBuffer
from moozi.parameter_optimizer import ParameterServer
from moozi.rollout_worker import RolloutWorker

from lib import (
    training_suite_factory,
    make_test_worker_universe,
    make_reanalyze_universe,
    make_env_worker_universe,
    config,
)


# %%
ps = ray.remote(num_gpus=config.param_opt.num_gpus)(ParameterServer).remote(
    training_suite_factory=training_suite_factory(config), use_remote=True
)
rb = ray.remote(ReplayBuffer).remote(**config.replay)

config.train.num_epochs = 30

# %%
train_workers = [
    ray.remote(num_gpus=config.train.env_worker.num_gpus)(RolloutWorker).remote(
        partial(make_env_worker_universe, config), name=f"rollout_worker_{i}"
    )
    for i in range(config.train.env_worker.num_workers)
]

# %%
jb_logger = JAXBoardLoggerRemote.remote()
terminal_logger = TerminalLoggerRemote.remote()
start_training = False
for w in train_workers:
    w.set.remote("params", ps.get_params.remote())
    w.set.remote("state", ps.get_state.remote())

for epoch in range(1, config.train.num_epochs + 1):
    logger.info(f"Epoch {epoch}")

    for w in train_workers:
        sample = w.run.remote()
        rb.add_trajs.remote(sample, from_env=True)

# %%
ray.get(rb.get_targets_size.remote())

# %%
for _ in range(600):
    ps.update.remote(
        rb.get_train_targets_batch.remote(config.train.batch_size),
        config.train.batch_size,
    )

# local version
# for _ in range(100):
#     print(
#         ps.update(
#             rb.get_train_targets_batch(config.train.batch_size),
#             config.train.batch_size,
#         )
#     )

# %%
ray.get(ps.save.remote())

# %%
